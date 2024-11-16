# streamlit_app.py
import streamlit as st
import logging
from main import IndexLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_app():
    """Initialize the app using the IndexLoader module."""
    try:
        loader = IndexLoader(db_path="./chroma_db")
        index = loader.load_index()
        query_engine = loader.create_query_engine(index)
        return query_engine
    except Exception as e:
        logger.error(f"Error initializing app: {str(e)}")
        raise

def main():
    st.set_page_config(
        page_title="Knowledge AI Agent",
        page_icon="\U0001F3EA",
        layout="wide"
    )

    st.title("Knowledge AI Agent")

    # Initialize system
    try:
        if 'query_engine' not in st.session_state:
            with st.spinner("Initializing system..."):
                st.session_state.query_engine = initialize_app()
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
                    response_text = str(response.response)

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
