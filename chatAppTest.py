# streamlit_app.py
import streamlit as st
from RAG.LLM import initialize_chatbot, create_query_engine

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def get_response_text(response):
    """Extract just the response text from the LlamaIndex response object"""
    # print(response.response)
    return str(response.response)

def main():
    st.set_page_config(
        page_title="Knowledge AI Agent",
        page_icon="🏪",
        layout="wide"
    )
    
    with st.sidebar:
        st.image("https://tile.loc.gov/image-services/iiif/service:ll:llscd:57026883:00010000/full/pct:100/0/default.jpg", use_container_width=True)
        st.header("Indian Constitution AI Agent")
        st.markdown(
            """
            - 💡 Ask questions related to knowledge domains.
            - 🛠 Powered by **Retrieval-Augmented Generation**.
            - 📚 Data source: Indian Constitution articles.
            """
        )
        st.divider()
        st.info("Developed using **Streamlit** and **LLM Technologies**.")
    
    # Title and Introduction
    st.title("📚 Indian Constitution AI Agent")
    st.write("Welcome! I can answer your queries related to knowledge domains using the Indian Constitution as the dataset. Start by asking a question below!")
    st.divider()
    # Initialize system
    try:
        if 'query_engine' not in st.session_state:
            with st.spinner("Initializing system..."):
                index = initialize_chatbot()
                st.session_state.query_engine = create_query_engine(index)
            st.success("System initialized successfully!")
        

        
        # Main chat interface
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").markdown(content)
        
        # Chat input
        if query := st.chat_input("Ask about products..."):
            st.chat_message("user").write(query)
            
            try:
                with st.spinner("Analyzing your question..."):
                    response = st.session_state.query_engine.query(query)
                    response_text = get_response_text(response)
                
                # Display the response
                st.chat_message("assistant").markdown(response_text)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_text}
                ])
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please try rephrasing your question.")
    
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.warning("Please check your configuration and try again.")

if __name__ == "__main__":
    main()