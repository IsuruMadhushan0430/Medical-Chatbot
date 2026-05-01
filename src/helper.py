import glob
import os
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document

def load_pdf_file(data):
    documents = []
    for file_path in glob.glob(os.path.join(data, "*.pdf")):
        print(f"Loading PDF: {file_path}")
        reader = PdfReader(file_path)
        loaded_documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            loaded_documents.append(
                Document(
                    page_content=page_text,
                    metadata={"source": file_path, "page": page_number},
                )
            )
        print(f"Loaded {len(loaded_documents)} pages from {file_path}")
        documents.extend(loaded_documents)

    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_split(minimal_docs):
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []

    for doc in minimal_docs:
        text = doc.page_content or ""
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=dict(doc.metadata),
                )
            )
            if end >= len(text):
                break
            start = end - chunk_overlap

    return chunks

def download_hugging_face_embeddings():
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings