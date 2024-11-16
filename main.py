import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding

class IndexLoader:
    def __init__(self, db_path, api_key_var="GOOGLE_API_KEY"):
        """Initialize the IndexLoader class.
        
        Args:
            db_path (str): Path to the persistent database for Chroma.
            api_key_var (str): Environment variable name for the API key. Default is 'GOOGLE_API_KEY'.
        """
        load_dotenv()
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
        
        # Load Chroma client and collection
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.chroma_collection = self.client.get_collection("quickstart")
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

    def load_index(self):
        """Load an existing vector store index from the vector store.
        
        Returns:
            VectorStoreIndex: The loaded index.
        """
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        return index

    def create_query_engine(self, index, prompt_template=None):
        """Create a query engine from the index.
        
        Args:
            index (VectorStoreIndex): The index to create a query engine from.
            prompt_template (str, optional): Custom prompt template for the query engine.
        
        Returns:
            QueryEngine: The configured query engine.
        """
        if prompt_template is None:
            template = (
                """ You are an assistant for question-answering tasks.
                Use the following context to answer the question.
                If you don't know the answer, just say that you don't know.
                Keep the answer concise.
                
                Question: {query_str} \nContext: {context_str} \nAnswer:"""
            )
            prompt_template = PromptTemplate(template)
        
        query_engine = index.as_chat_engine(text_qa_template=prompt_template)
        return query_engine

# Example usage (this part should be in your main script):
# if __name__ == "__main__":
#     loader = IndexLoader(db_path="./chroma_db")
#     index = loader.load_index()
#     engine = loader.create_query_engine(index)
#     while True:
#         query = input("Enter your question: ")
#         response = engine.query(query)
#         print(response)
#         print("\n")
