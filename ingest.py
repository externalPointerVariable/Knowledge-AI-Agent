import os
import re
import chromadb
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
load_dotenv()


llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
embed_model = GeminiEmbedding(model_name="models/embedding-001")

llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.llm = llm
Settings.embed_model = embed_model

# Load data from PDF
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("data").load_data()

# Create a client and a new collection
client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("quickstart")

# Create a vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Create an index from the documents and save it to the disk.
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,show_progress=True,
)