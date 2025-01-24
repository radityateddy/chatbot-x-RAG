import os
import json
import time
from hashlib import md5
import chromadb
import chromadb.config
import pypdf
import ollama
import streamlit as st

# Inisialisasi klien ChromaDB dan koleksi
chroma_client = chromadb.Client(chromadb.config.Settings(allow_reset=True))
collection = chroma_client.get_or_create_collection("context")

# Lokasi folder dokumen dan metadata
DOCS_FOLDER = "docs"
METADATA_FILE = "metadata.json"

def load_metadata():
    """Memuat metadata dari file JSON."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as file:
            return json.load(file)
    return {}

def save_metadata(metadata):
    """Menyimpan metadata ke file JSON."""
    with open(METADATA_FILE, "w") as file:
        json.dump(metadata, file, indent=4)

def file_hash(file_path):
    """Menghitung hash dari file."""
    with open(file_path, "rb") as file:
        return md5(file.read()).hexdigest()

def clean_text(text):
    """Membersihkan teks dari noise."""
    return ' '.join(text.split())  # Menghapus spasi berlebih

def clear_embeddings_and_metadata():
    """Menghapus semua embedding dan metadata."""
    print("Menghapus semua embedding dari koleksi dan metadata.json...")
    try:
        # Uncomment if you don't want to reset the collection
        # chroma_client.reset()
        print("Semua embedding telah dihapus dari koleksi.")
    except Exception as e:
        print(f"Error saat menghapus embedding: {e}")

    # Hapus file metadata.json jika ada
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
        print(f"File metadata {METADATA_FILE} telah dihapus.")
    else:
        print(f"File metadata {METADATA_FILE} tidak ditemukan.")


def regenerate_embeddings():
    """Memproses ulang embedding dari dokumen."""
    metadata = load_metadata()
    new_metadata = {}
    files_changed = False

    for file_name in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, file_name)
        if file_name.endswith(".pdf"):
            current_hash = file_hash(file_path)
            last_hash = metadata.get(file_name, {}).get("hash")

            # Proses ulang jika hash berubah
            if current_hash != last_hash:
                process_pdf(file_path)
                files_changed = True

            new_metadata[file_name] = {
                "hash": current_hash,
                "modified_time": os.path.getmtime(file_path),
            }

    # Hapus embedding untuk file yang dihapus
    for file_name in metadata:
        if file_name not in new_metadata:
            all_data = collection.get()
            ids_to_delete = [
                item["id"] for item in all_data.get("metadatas", []) if item["id"].startswith(file_name)
            ]
            if ids_to_delete:
                collection.delete(where={"id": {"$in": ids_to_delete}})
            files_changed = True

    if files_changed:
        save_metadata(new_metadata)

def process_pdf(file_path):
    """Menghasilkan embedding dari PDF."""
    with open(file_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages):
            content = clean_text(page.extract_text())
            if content.strip():  # Abaikan halaman kosong
                collection.add(
                    documents=[content],
                    ids=[f"{os.path.basename(file_path)}-page-{page_num}"]
                )

# Pilihan: Hapus semua data sebelum regenerasi embedding
clear_embeddings_and_metadata()

# Jalankan pembaruan embedding
regenerate_embeddings()

# Query input dari pengguna
query = input(">>> ")

# Query dari koleksi
closestPages = collection.query(
    query_texts=[query],
    n_results=3
)

# Format hasil embedding menjadi konteks
context = "\n".join(
    f"Page {i+1}: {doc}" for i, doc in enumerate(closestPages["documents"])
)

# Persiapkan pesan untuk LLM
messages = [
    {
        "role": "system",
        "content": (
            "Berikut adalah informasi relevan yang diambil dari dokumen. "
            "Jawab pertanyaan pengguna dengan fokus pada konteks berikut:"
        ),
    },
    {
        "role": "system",
        "content": context,
    },
    {
        "role": "user",
        "content": query,
    },
]

# Kirim pesan ke model Ollama
response = ollama.chat(
    model="llama3:latest",
    messages=messages
)

# Tampilkan respons dari model
print("\nModel Response:")
print(response["message"]["content"])
