# vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv("./f1_complete_data.csv")
# Load embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Folder for vector store
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Simplified descriptive page_content
        page_content = (
            f"{row['Driver']} finished {row['Position']} at the {row['Track']} Grand Prix "
            f"in {row['Year']} with team {row['Team']}."
        )
        # Store all other fields as metadata
        metadata = {key: value for key, value in row.items() if key not in ["Track", "Position", "Driver", "Team", "Year"]}
        doc = Document(page_content=page_content, metadata=metadata, id=str(i))
        
        documents.append(doc)
        ids.append(str(i))
    
    logger.info(f"Created {len(documents)} documents for vector store.")

# Initialize vector store
vector_store = Chroma(
    collection_name="f1_reviews",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    # Add documents to vector store
    vector_store.add_documents(documents, ids=ids)
    logger.info(f"Added {len(documents)} documents to vector store.")

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
logger.info("Retriever initialized.")