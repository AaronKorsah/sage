import os
import logging
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from groq import Groq
from sentence_transformers import SentenceTransformer

from smart_chunker import smart_chunk

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = genai.Client()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed(texts):
    vectors = model.encode(texts)
    return vectors.tolist()


def read_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        logger.error(f"File '{pdf_path}' not found.")
        raise FileNotFoundError(pdf_path)
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    logger.info(f"Read {len(text)} characters from {len(reader.pages)} page(s)")
    return text


def ingest(pdf_path, collection):
    text = read_pdf(pdf_path)
    chunks = smart_chunk(text, chunk_size=500, overlap=100)
    logger.info(f"Created {len(chunks)} chunks")

    vectors = embed(chunks)

    filename = os.path.basename(pdf_path)

    collection.add(
        ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": filename} for _ in chunks]
    )
    logger.info(f"Stored {len(chunks)} chunks from {filename}")


def retrieve(question, collection, source=None, n_results=3):
    question_vector = embed([question])[0]

    query_kwargs = {
        "query_embeddings": [question_vector],
        "n_results": n_results
    }

    if source:
        query_kwargs["where"] = {"source": source}

    results = collection.query(**query_kwargs)
    return results["documents"][0]


def ask_llm(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.warning(f"Gemini failed ({e}). Falling back to Groq...")
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content