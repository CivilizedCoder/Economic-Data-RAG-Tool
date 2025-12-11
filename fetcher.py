#!/usr/bin/env python3

import requests
import os
import pickle
from datetime import datetime, timedelta
from slugify import slugify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Configuration ---
API_KEY = "690a37d2df41a6.40496773" # its free, I don't care, lol
NEWS_DIR = "news_embeddings"
API_URL = "https://eodhistoricaldata.com/api/news"
DAYS_TO_FETCH = 7
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
# ---------------------

def setup_model():
    """Loads the embedding model."""
    print(f"Loading model {MODEL_NAME}...")
    try:
        return SentenceTransformer(MODEL_NAME, device='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_article_to_data(title, content, source, date, model):
    """
    Splits article into chunks and creates embeddings.
    Returns a dictionary containing chunks, embeddings, and metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    full_text = f"Title: {title}\nSource: {source}\nDate: {date}\n\n{content}"
    chunks = text_splitter.split_text(full_text)

    if not chunks:
        return None


    filtered_chunks = []
    for chunk in chunks:
        clean_chunk = chunk.strip()

        if clean_chunk.startswith("Title:") or clean_chunk.endswith("+00:00"):
            continue

        filtered_chunks.append(chunk)

    if not filtered_chunks:
        return None

    embeddings = model.encode(filtered_chunks, normalize_embeddings=True)

    data_package = {
        "title": title,
        "chunks": filtered_chunks,
        "embeddings": embeddings,
        "source": source,
        "date": date
    }
    return data_package

def fetch_and_save_embeddings():
   
    os.makedirs(NEWS_DIR, exist_ok=True)

   
    model = setup_model()
    if not model:
        return

    
    start_date = (datetime.now() - timedelta(days=DAYS_TO_FETCH)).strftime('%Y-%m-%d')
    print(f"Fetching news from {start_date}...")

    params = {
        'api_token': API_KEY,
        't': 'Economy',
        'from': start_date,
        'fmt': 'json'
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        articles = response.json()

        if not articles:
            print("No new articles found.")
            return

        saved_count = 0
        skipped_count = 0

        print(f"Processing {len(articles)} articles...")

        for article in articles:
            content = article.get('content')
            title = article.get('title')

            if not content or not title:
                continue

            
            safe_filename = slugify(title) + ".pkl"
            file_path = os.path.join(NEWS_DIR, safe_filename)

            
            if not os.path.exists(file_path):
                try:
                    # Process (Chunk & Embed)
                    data = process_article_to_data(
                        title,
                        content,
                        article.get('source'),
                        article.get('date'),
                        model
                    )

                    if data:
                        # Save to filesystem using pickle
                        with open(file_path, 'wb') as f:
                            pickle.dump(data, f)

                        print(f"Saved embeddings for: {title[:30]}...")
                        saved_count += 1
                except Exception as e:
                    print(f"Error processing {title}: {e}")
            else:
                skipped_count += 1

        print("\n--- Fetch & Embed Complete ---")
        print(f"New files created: {saved_count}")
        print(f"Skipped (already exist): {skipped_count}")

    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    fetch_and_save_embeddings()
