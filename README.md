
# PDF RAG Chatbot

A chatbot that lets you upload a PDF and ask questions about it. 
Built using LangGraph, FAISS, LangChain, and Groq's LLaMA model.

## What it does
- Takes any PDF as input
- Splits it into chunks and stores them in a FAISS vector database
- When you ask a question, it retrieves relevant chunks and generates an answer using Groq's LLaMA-3.3-70B model
- The pipeline is coordinated using a LangGraph state machine

## Tech Stack
- LangChain — for chaining the retrieval and generation steps
- LangGraph — for managing the pipeline as a state graph
- FAISS — for local vector storage and similarity search
- HuggingFace `all-MiniLM-L6-v2` — for text embeddings
- Groq API — for fast LLM inference
- Streamlit — for the UI

## How to run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key in a `.env` file
4. Run: `streamlit run app.py`
