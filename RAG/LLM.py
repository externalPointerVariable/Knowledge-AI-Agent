import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_chatbot(db="pinecone"):
    """Initialize the chatbot with the latest OpenAI models"""
    try:
        load_dotenv()
        
        llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"],model="models/gemini-1.5-flash-8b")
        embed_model = GeminiEmbedding(model_name="models/embedding-001")
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        
        if db == "pinecone":
            pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            pinecone_index = pinecone_client.Index("indianconstitution")
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        else:
            client = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = client.get_or_create_collection("constitution")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


        index = VectorStoreIndex.from_vector_store(vector_store)
        
        return index
    
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

def create_query_engine(index):
    
    DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
    )
    DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
        DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    )


    return index.as_query_engine(
        text_qa_template=DEFAULT_TEXT_QA_PROMPT,
        similarity_top_k=7,
    )

def ingestion_pipeline(db="pinecone"):
    load_dotenv()
    llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"],model="models/gemini-1.5-pro-002")
    embed_model = GeminiEmbedding(model_name="models/embedding-001")
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    # Load data from PDF
    documents = SimpleDirectoryReader("data").load_data()

    if db == "pinecone":
        pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        pinecone_index = pinecone_client.Index("indianconstitution")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    else:
    # Create a client and a new collection
        client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = client.get_or_create_collection("constitution")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
    # Create a storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Create an index from the documents and save it to the disk.
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,show_progress=True,
    )