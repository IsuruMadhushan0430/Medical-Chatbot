from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.helper import (
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    load_pdf_file,
    text_split,
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created Pinecone index: {index_name}")
else:
    print(f"Pinecone index already exists: {index_name}")

index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

BATCH_SIZE = 100

print("Loading PDF documents...")
extracted_data = load_pdf_file(data="data/")
print(f"Loaded {len(extracted_data)} documents")

print("Filtering documents...")
filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting text into chunks...")
texts_chunk = text_split(filter_data)
print(f"Created {len(texts_chunk)} chunks")

print("Loading embedding model...")
embeddings = download_hugging_face_embeddings()

print("Generating embeddings...")
chunk_texts = [doc.page_content for doc in texts_chunk]
chunk_vectors = embeddings.embed_documents(chunk_texts)

print("Upserting vectors to Pinecone...")
vectors = []
for position, (doc, vector) in enumerate(zip(texts_chunk, chunk_vectors), start=1):
    vectors.append(
        (
            f"{index_name}-{position}",
            vector,
            {
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content,
            },
        )
    )

for start in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[start : start + BATCH_SIZE]
    index.upsert(vectors=batch)
    print(f"Upserted vectors {start + 1}-{start + len(batch)}")

stats = index.describe_index_stats()
print(f"Stored records in Pinecone index '{index_name}': {stats.total_vector_count}")
