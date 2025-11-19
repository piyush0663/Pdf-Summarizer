import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, shutil
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# -----------------------------
# EXTRACT PDF TEXT
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# -----------------------------
# TEXT SPLITTING
# -----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )
    return splitter.split_text(text)


# -----------------------------
# CREATE VECTOR STORE
# -----------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# -----------------------------
# CONVERSATIONAL CHAIN (LLM)
# -----------------------------
def get_conversational_chain():

    prompt_template = """
    You are an AI assistant for answering questions from PDFs.
    Use ONLY the provided context. Do NOT make up answers.

    If the answer is not found in the context, reply:
    "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro-latest",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# -----------------------------
# HANDLE USER QUERY
# -----------------------------
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.write("### Reply:")
        st.write(response["output_text"])

        # Store conversation in memory
        st.session_state.history.append(f"**You:** {user_question}")
        st.session_state.history.append(f"**Bot:** {response['output_text']}")

    except Exception as e:
        st.error(f"Error: {e}")


# -----------------------------
# MAIN STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config("Multi PDF Chatbot", page_icon="📚")
    st.header("📚 Multi-PDF Chat Agent 🤖")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    user_question = st.text_input(
        "Ask any question from your uploaded PDF files:"
    )

    if user_question:
        user_input(user_question)

    # Display conversation history
    for msg in st.session_state.history:
        st.markdown(msg)

    # Sidebar PDF handling
    with st.sidebar:
        st.title("📁 Upload PDF Files")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        # Show PDF file names
        if pdf_docs:
            st.subheader("Uploaded Files:")
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")

        # Process button
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDF documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed and vector store created!")
            else:
                st.warning("Please upload at least one PDF file.")

        # Clear vector index
        if st.button("Clear Previous Index"):
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
                st.success("Vector index cleared!")
            else:
                st.info("No index found to clear.")


if __name__ == "__main__":
    main()
