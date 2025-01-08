from RAG.LLM import ingestion_pipeline

if __name__ == "__main__":
    db = input("Enter the name of the database you want to use (pinecone/chromadb): ")
    ingestion_pipeline(db)

