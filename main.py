import requests
from bs4 import BeautifulSoup
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from openai import OpenAI
import openai

client = OpenAI(api_key='your API key')
import streamlit as st

# Constants
DB_PATH = 'newmanufacturing_data_new.db'
URL = 'https://economictimes.indiatimes.com/industry/indl-goods/svs/engineering'


# Scraping Articles from URL
def scrape_articles(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        for article in soup.find_all('div', class_='eachStory'):
            title = article.find('h3').get_text(strip=True)
            link = 'https://economictimes.indiatimes.com' + article.find('a')['href']
            articles.append({'title': title, 'link': link})
        return articles
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []


# Main workflow to set up FTS database and insert articles
def main_fts():
    # Step 1: Scrape the articles
    url = 'https://economictimes.indiatimes.com/industry/indl-goods/svs/engineering'
    articles = scrape_articles(url)

    # Step 2: Set up the SQLite database with FTS5
    conn, cursor = setup_database_with_fts()

    # Step 3: Insert articles into the FTS5 table
    insert_articles_fts(cursor, articles)

    # Commit and close the connection
    conn.commit()
    conn.close()

    print("FTS table creation and insertion completed!")


# Extracting Content from Article URL
def get_article_content(url):
    try:
        article_response = requests.get(url)
        article_soup = BeautifulSoup(article_response.content, 'html.parser')
        return article_soup.find('div', class_='artText').get_text(strip=True)
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None


# Function to set up the SQLite database with FTS5 support
def setup_database_with_fts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create an FTS5 virtual table for full-text search
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
        title, 
        link, 
        content
    )
    ''')
    conn.commit()
    return conn, cursor


# Insert Articles with FTS5 Support
def insert_articles_fts(cursor, articles):
    for article in articles:
        content = get_article_content(article['link'])
        if content:
            cursor.execute('''
            INSERT INTO articles_fts (title, link, content) VALUES (?, ?, ?)
            ''', (article['title'], article['link'], content))


# Sentence-BERT Embedding for Articles
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')


def generate_embedding(model, content):
    return model.encode(content)


# Store Embeddings in SQLite
def create_embeddings_table(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS article_embeddings (
        article_id INTEGER PRIMARY KEY,
        embedding BLOB
    )
    ''')


def store_embedding(cursor, article_id, embedding):
    embedding_blob = pickle.dumps(embedding)
    cursor.execute('''
    INSERT INTO article_embeddings (article_id, embedding) VALUES (?, ?)
    ''', (article_id, embedding_blob))


# Generate and Store Article Embeddings
def generate_and_store_embeddings():
    model = load_model()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    create_embeddings_table(cursor)
    cursor.execute('SELECT rowid, content FROM articles_fts')
    articles = cursor.fetchall()

    for article_id, content in articles:
        if content:
            embedding = generate_embedding(model, content)
            store_embedding(cursor, article_id, embedding)

    conn.commit()
    conn.close()


# Search for Similar Articles Using Embeddings
def find_similar_articles(query, top_n=5):
    model = load_model()
    query_embedding = model.encode(query)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT article_id, embedding FROM article_embeddings')
    embeddings = cursor.fetchall()

    similar_articles = []
    for article_id, embedding_blob in embeddings:
        stored_embedding = pickle.loads(embedding_blob)
        similarity = 1 - cosine(query_embedding, stored_embedding)
        similar_articles.append((article_id, similarity))

    similar_articles.sort(key=lambda x: x[1], reverse=True)
    return similar_articles[:top_n]


# OpenAI GPT-4 for Response Generation

# Replace with your OpenAI API key


def generate_response(prompt):
    response = client.completions.create(engine="text-davinci-003",  # Use text-davinci-003 for better performance
    prompt=prompt,
    max_tokens=150)
    return response.choices[0].text.strip()


# def summarize_article(content):
#     prompt = f"Summarize the following article:\n\n{content}"
#     response = client.completions.create(engine="text-davinci-003", prompt=prompt, max_tokens=100)
#     return response.choices[0].text.strip()
def summarize_article(content):
    prompt = f"Summarize the following article:\n\n{content}"
    response = openai.Completion.create(  # Use openai.Completion.create (capital C)
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()


# Retrieve Relevant Articles Using FTS5
def retrieve_relevant_articles(query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT title, content FROM articles_fts WHERE articles_fts MATCH ?", (query,))
    return cursor.fetchall()


# Retrieval-Augmented Generation (RAG) Pipeline
def rag_pipeline(user_query):
    articles = retrieve_relevant_articles(user_query)
    if not articles:
        return "No relevant articles found."

    prompt = f"User query: {user_query}\n\n"
    prompt += "Here are some relevant articles:\n"
    for title, content in articles:
        summary = summarize_article(content)
        prompt += f"Title: {title}\nSummary: {summary}\n\n"

    return generate_response(prompt)

main_fts()
# Streamlit Interface
st.title("Manufacturing and Supply Chain Chatbot")
user_query = st.text_input("Ask me anything about manufacturing and supply chain:")

if user_query:
    response = rag_pipeline(user_query)
    st.write(response)
