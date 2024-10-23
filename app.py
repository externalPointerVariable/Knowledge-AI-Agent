import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex
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

# Load from disk
load_client = chromadb.PersistentClient(path="./chroma_db")

# Fetch the collection
chroma_collection = load_client.get_collection("quickstart")

# Fetch the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Get the index from the vector store
index = VectorStoreIndex.from_vector_store(
    vector_store
)

from llama_index.core import PromptTemplate

template = (
    """ You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {query_str} \nContext: {context_str} \nAnswer:"""
)
llm_prompt = PromptTemplate(template)
query_engine = index.as_chat_engine(text_qa_template=llm_prompt)
while(True):
    query = input("Enter your question: ")
    response = query_engine.query(query)
    print(response)
    print("\n")