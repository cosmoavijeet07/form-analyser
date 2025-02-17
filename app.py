import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from PyPDF2 import PdfReader
import os
import time
import json
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from pydantic import Field, BaseModel

# Load environment variables
load_dotenv(Path(".env"))

# Configure Streamlit UI
st.set_page_config(page_title="Call Transcript Analyzer", layout="wide")
st.title("Call Transcript Analyzer & Q&A Bot")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "faiss_vector_index" not in st.session_state:
    st.session_state.faiss_vector_index = None

if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = {}

# Custom Gemini LLM class
class GeminiLLM(LLM, BaseModel):
    model_name: str = Field(default="gemini-1.5-flash", description="Gemini Model Name")
    model: Optional[Any] = Field(None, description="Gemini Model Instance")

    def __init__(self, model_name: str, **data):
        super().__init__(model_name=model_name, **data)
        self.model = genai.GenerativeModel(model_name=self.model_name)

    def _call(self, transcript: str, stop: Optional[List[str]] = None) -> str:
        system_prompt = """
        You are an AI assistant designed to extract structured information from call transcripts. 
        Analyze the transcript and return a JSON object with the following fields:
        - "name": (Full name of the customer, or "Unknown" if not mentioned)
        - "age": (Age of the customer, or "Unknown" if not mentioned)
        - "sentiment": (Overall sentiment of the customer - "Positive", "Negative", or "Neutral")
        - "issue_summary": (Brief summary of the customer's issue)
        - "call_duration": (Approximate duration of the call in minutes, if available)
        - "agent_name": (The name of the support agent, if mentioned)
        
        Ensure the response is strictly formatted as JSON with **no extra text**.
        """

        full_prompt = f"{system_prompt}\n\nCall Transcript:\n{transcript}\n\nExtracted Info:"
        response = self.model.generate_content(full_prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

# Function to extract text from uploaded files
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    else:
        st.error("Unsupported file format. Please upload a PDF or TXT file.")
        return None
    return text

# Function to extract details using Gemini AI
def extract_details_with_gemini(text):
    gemini_llm = GeminiLLM(model_name='gemini-1.5-flash')
    response = gemini_llm._call(text)
    
    # Extract structured data from Gemini response
    try:
        extracted_info = json.loads(response)  # Convert JSON string to dictionary
        return extracted_info
    except json.JSONDecodeError:
        st.error("Error in extracting details using Gemini AI. Please try again.")
        return {}

# Function to process uploaded transcript
def process_uploaded_transcript(uploaded_file):
    text = extract_text(uploaded_file)
    if not text:
        return

    extracted_info = extract_details_with_gemini(text)
    
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

    embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_vector_store = FAISS.from_texts([text], embedding_function)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    faiss_vector_store.add_texts(texts[:50])

    st.session_state.faiss_vector_index = VectorStoreIndexWrapper(vectorstore=faiss_vector_store)
    st.session_state.pdf_processed = True
    st.session_state.extracted_info = extracted_info

    st.success("Transcript processed successfully!")

# Sidebar for file upload
st.sidebar.markdown("## Upload a Call Transcript")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
if uploaded_file:
    process_uploaded_transcript(uploaded_file)

# Display extracted information in a form
if st.session_state.get("pdf_processed", False):
    extracted_info = st.session_state.get("extracted_info", {})
    
    st.sidebar.markdown("## Auto-Filled BPO Form")
    with st.sidebar.form("bpo_form"):
        st.text_input("Customer Name", value=extracted_info.get("name", "Unknown"))
        st.text_input("Customer Age", value=extracted_info.get("age", "Unknown"))
        st.text_input("Sentiment", value=extracted_info.get("sentiment", "Unknown"))
        st.text_area("Issue Summary", value=extracted_info.get("issue_summary", "Unknown"))
        st.text_input("Call Duration (minutes)", value=extracted_info.get("call_duration", "Unknown"))
        st.text_input("Agent Name", value=extracted_info.get("agent_name", "Unknown"))
        st.form_submit_button("Submit")

# Typing animation
def typing_animation(text, speed):
    for char in text:
        yield char
        time.sleep(speed)

if "intro_displayed" not in st.session_state:
    st.session_state.intro_displayed = True
    st.write_stream(typing_animation("Hello, I am Docs, your Call Transcript Assistant!", 0.02))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chatbot interaction
prompt = st.chat_input("Ask about the transcript..")

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    query_text = prompt.strip()
    gemini_llm = GeminiLLM(model_name='gemini-1.5-flash')

    if st.session_state.faiss_vector_index:
        answer = st.session_state.faiss_vector_index.query(query_text, llm=gemini_llm).strip()
        with st.chat_message("assistant"):
            st.write_stream(typing_animation(answer, 0.02))
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Transcript not processed. Upload and process a file first.")
