import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding

class IndexBuilder:
    def __init__(self, data_path, db_path, api_key_var="GOOGLE_API_KEY"):
        """Initialize the IndexBuilder class.
        
        Args:
            data_path (str): Path to the directory containing data files.
            db_path (str): Path to the persistent database for Chroma.
            api_key_var (str): Environment variable name for the API key. Default is 'GOOGLE_API_KEY'.
        """
        load_dotenv()
        self.data_path = data_path
        self.db_path = db_path
        self.api_key = os.environ.get(api_key_var)
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_var} is not set.")
        
        # Initialize LLM and embedding model
        self.llm = Gemini(api_key=self.api_key)
        self.embed_model = GeminiEmbedding(model_name="models/embedding-001")
        
        # Apply settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Create a Chroma client and collection
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.chroma_collection = self.client.get_or_create_collection("quickstart")
        
        # Initialize vector store and storage context
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def load_documents(self):
        """Load documents from the specified data path."""
        reader = SimpleDirectoryReader(self.data_path)
        documents = reader.load_data()
        return documents

    def create_index(self, show_progress=True):
        """Create a vector store index from documents and save it to disk.
        
        Args:
            show_progress (bool): Whether to display progress while creating the index.
        
        Returns:
            VectorStoreIndex: The created index.
        """
        documents = self.load_documents()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context, show_progress=show_progress
        )
        return index


