import os
from dotenv import load_dotenv
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
import streamlit as st

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_answer(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
    You are a helpful assistant. Answer the question using the PDF context below.
    You also have access to the recent conversation history to understand follow-up questions.
    If the answer is not in the context, say "I could not find the answer in this PDF."

    PDF Context:
    {context}

    Conversation History:
    {history_text}

    Current Question:
    {user_question}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ─── Streamlit UI ───────────────────────────────────────────

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

st.title("📄 Chat with your PDF")
st.markdown("Upload a PDF and ask anything about it.")

st.markdown("""
<style>
[data-testid="stChatInput"] {
    border-color: #4a4a4a !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 1px #a855f7 !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Reading and processing your PDF..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                chunks = get_text_chunks(raw_text)
                create_vector_store(chunks)
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
            st.success(f"✅ Done! {len(chunks)} chunks created.")

    if st.session_state.get("pdf_processed"):
        st.info(f"📄 Active: {st.session_state.pdf_name}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

_disabled = not st.session_state.get("pdf_processed", False)
if prompt := st.chat_input("Ask a question about your PDF...", disabled=_disabled):
    if not st.session_state.get("pdf_processed"):
        st.warning("⚠️ Please upload and process a PDF first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(prompt, st.session_state.messages)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
