import os
import streamlit as st
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

DB_DIR = "faiss_index"

class ChatState(TypedDict):
    question: str
    context: str
    answer: str

@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = "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"
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=api_key
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_uploaded_pdf(uploaded_file):
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(temp_filename)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save index to local disk storage
    vector_store.save_local(DB_DIR)
    
    os.remove(temp_filename)
    return vector_store

def retrieve_docs(state: ChatState):
    question = state["question"]
    vector_store = st.session_state["vector_store"]
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate_answer(state: ChatState):
    context = state["context"]
    question = state["question"]
    
    prompt = f"""
    You are an expert AI assistant answering questions based strictly on the provided PDF context.
    If the answer cannot be found in the context, say "I cannot find the answer in the provided document."
    
    Context:
    {context}
    
    Question: {question}
    Answer:
    """
    llm = get_llm()
    response = llm.invoke(prompt)
    return {"answer": response.content}

def create_rag_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

# --- STREAMLIT ORIGINAL MINIMAL USER INTERFACE ---
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG-Based PDF Chatbot")
st.subheader("Upload any document and ask questions instantly")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Automatically load the database from your Mac if it already exists
if "vector_store" not in st.session_state:
    if os.path.exists(DB_DIR):
        st.session_state["vector_store"] = FAISS.load_local(
            DB_DIR, get_embeddings(), allow_dangerous_deserialization=True
        )
    else:
        st.session_state["vector_store"] = None

with st.sidebar:
    st.header("Document Center")
    uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing PDF and building vector database..."):
            st.session_state["vector_store"] = process_uploaded_pdf(uploaded_file)
        st.success("PDF fully processed and loaded into FAISS!")
    elif st.session_state["vector_store"] is not None:
        st.info("Stored document index active.")

if st.session_state["vector_store"] is not None:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("Ask me anything about your PDF..."):
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                graph_app = create_rag_graph()
                result = graph_app.invoke({"question": user_query})
                ai_response = result["answer"]
                st.markdown(ai_response)
                
        st.session_state["messages"].append({"role": "assistant", "content": ai_response})
else:
    st.warning("Please upload a PDF file in the sidebar to kick off the chatbot application.")