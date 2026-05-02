# Medical Chatbot

A retrieval-augmented medical chatbot built with Flask, Pinecone, Hugging Face embeddings, and Google Gemini. The app loads PDF documents from `data/`, splits them into chunks, stores embeddings in Pinecone, and serves a simple chat UI through a Flask web app.

## What It Does

- Loads medical PDFs from the `data/` folder.
- Splits documents into overlapping text chunks.
- Generates 384-dimensional embeddings with `BAAI/bge-small-en-v1.5`.
- Stores chunk embeddings and source text in Pinecone.
- Retrieves the most relevant chunks for each question.
- Uses `gemini-2.5-flash` to generate answers from retrieved context.

## Project Structure

- `app.py` - Flask app that serves the chat UI and handles chat requests.
- `store_index.py` - builds the Pinecone index from local PDFs.
- `src/helper.py` - document loading, filtering, chunking, and embedding helpers.
- `src/prompt.py` - system prompt for the retrieval chain.
- `templates/chat.html` - browser chat interface.
- `data/` - place source PDF files here.

## Requirements

- Python 3.10 or newer
- A Pinecone account and API key
- A Google Gemini API key

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root with the following values:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Build The Pinecone Index

Before starting the chatbot, load your PDFs into Pinecone:

```bash
python store_index.py
```

This script:

- creates the `medical-chatbot` Pinecone index if it does not already exist,
- loads all PDF files from `data/`,
- splits them into chunks,
- embeds the chunks, and
- upserts them to Pinecone in batches.

If you add or replace PDFs, run `store_index.py` again to refresh the index.

## Run The App

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://localhost:8080
```

## How The Chat Flow Works

1. The browser sends the user message to the `/get` route.
2. The app retrieves the top matching chunks from Pinecone.
3. The retrieved chunks are passed to the Gemini model with the medical system prompt.
4. The generated answer is returned to the browser.

## Updating The Knowledge Base

To change the chatbot knowledge base:

1. Add or remove PDF files in `data/`.
2. Run `python store_index.py` again.
3. Restart `python app.py` if the app is already running.

## Troubleshooting

- `ModuleNotFoundError` or import errors usually mean the virtual environment is not activated or dependencies are not installed.
- If Pinecone rejects an upsert because the payload is too large, lower the batch size in `store_index.py`.
- If the app starts but answers are empty or irrelevant, verify that the Pinecone index has been populated and that your PDFs contain extractable text.
- If Gemini requests fail, confirm that `GOOGLE_API_KEY` is set correctly.

## Notes

- The app currently uses a retrieval chain with `langchain_classic` imports.
- The Pinecone vector store is read from an existing index at runtime.
- The chat UI is rendered from `templates/chat.html`.