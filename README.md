# Knowledge-AI-Agent

Knowledge-AI-Agent is an AI-powered chatbot designed to answer queries related to the Indian Constitution. It leverages Retrieval-Augmented Generation (RAG) and LLM technologies to provide accurate and informative responses based on the dataset of the Indian Constitution articles.

## Features

- 💡 Ask questions related to knowledge domains.
- 🛠 Powered by **Retrieval-Augmented Generation**.
- 📚 Data source: Indian Constitution articles.
- Developed using **Streamlit** and **LLM Technologies**.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/externalPointerVariable/Knowledge-AI-Agent
    cd Knowledge-AI-Agent
    ```

2. Set up the environment file namely `.env` and set the following variables:
    ```env
    GOOGLE_API_KEY = // API key of google gemini
    PINECONE_API_KEY = // API key of PinconeDB
    PINECONE_INDEX = // Name of the index in pineconeDB
    ```

3. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    If using conda:
    ```sh
    conda create -n myenv
    conda activate myenv
    ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Create embedding of the data provided:
    ```sh
    python run ingestion.py
    ```

2. Run the Streamlit application:
    ```sh
    streamlit run deploy.py
    ```

3. Open your web browser and navigate to the provided URL (usually `http://localhost:8501`).

4. Interact with the AI agent by asking questions related to the Indian Constitution.


## License

This project is licensed under the MIT License. See the  file for more details.

## Acknowledgements

- Streamlit for providing an easy-to-use framework for building web applications.
- LLM Technologies for powering the AI capabilities of the chatbot.