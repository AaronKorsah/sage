import sys
import os
import logging
import requests
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

from prompt_builder import build_prompt

# ---------- SETUP ----------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = genai.Client()

# ---------- EMBEDDING ----------
def embed(texts, batch_size=32):
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = requests.post(
            "http://localhost:8001/embed",
            json={"inputs": batch}
        )
        response.raise_for_status()
        all_vectors.extend(response.json())
    return all_vectors

# ---------- PDF ----------
def read_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        logger.error(f"File '{pdf_path}' not found.")
        sys.exit(1)
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    logger.info(f"Read {len(text)} characters from {len(reader.pages)} page(s)")
    return text

# ---------- CHUNKING ----------
def chunk_text(text, chunk_size=200, overlap=40):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# ---------- INGEST ----------
def ingest(pdf_path, collection):
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)
    logger.info(f"Created {len(chunks)} chunks")

    vectors = embed(chunks)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors
    )
    logger.info(f"Stored {collection.count()} chunks")

# ---------- RETRIEVE ----------
def retrieve(question, collection, n_results=3):
    question_vector = embed([question])[0]
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=n_results
    )
    return results["documents"][0]

# ---------- ANSWER ----------
def ask_llm(prompt):
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

# ---------- MAIN ----------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python sage.py "<question>" <file.pdf>')
        sys.exit(1)

    question = sys.argv[1]
    pdf_path = sys.argv[2]

    # ChromaDB setup
    db = chromadb.PersistentClient(path="./chroma_db")
    try:
        db.delete_collection(name="sage_docs")
    except Exception:
        pass
    collection = db.get_or_create_collection(name="sage_docs")

    # Ingest
    ingest(pdf_path, collection)

    # Retrieve
    logger.info(f"Question: {question}")
    chunks = retrieve(question, collection)

    # Build prompt and ask
    prompt = build_prompt(question, chunks)
    answer = ask_llm(prompt)

    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print(f"\n{answer}\n")