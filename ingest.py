import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Use a standard local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_corpus(file_path="corpus.jsonl"):
    print(f"Loading corpus from {file_path}...")
    documents = []
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            # Create LangChain Document with metadata for filtering
            doc = Document(
                page_content=item['body'],
                metadata={
                    "country": item['country'],
                    "language": item['language'],
                    "type": item['type'],
                    "content_id": item['content_id']
                }
            )
            documents.append(doc)
    
    # Build and save FAISS index
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("faiss_index")
    print(f"Successfully ingested {len(documents)} items into 'faiss_index' folder.")

if __name__ == "__main__":
    ingest_corpus()