# 📚 Multi-PDF Chatbot using Google Gemini, LangChain & FAISS

A powerful **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **Gemini 1.5**, **LangChain**, **FAISS**, and **PyPDF2**.  
This app allows users to upload multiple PDF files, process them into vector embeddings, and ask questions based on their content with accurate, context-grounded answers.

---
<img width="1707" height="650" alt="Screenshot 2025-11-19 171641" src="https://github.com/user-attachments/assets/49b9cd65-eb3f-4351-a14e-d1aeab1234f9" />



## 🚀 Features

- ✔ Upload multiple PDF files  
- ✔ Extract text from PDFs  
- ✔ Intelligent chunking for better retrieval  
- ✔ Google Generative AI Embeddings  
- ✔ Vector search powered by FAISS  
- ✔ Chat with your PDFs using Gemini  
- ✔ Chat history memory  
- ✔ PDF file preview  
- ✔ Clear FAISS index option  
- ✔ Clean Streamlit UI  

---

## 🧠 Technology Stack

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| PDF Processing | PyPDF2 |
| Text Chunking | LangChain Text Splitter |
| Embeddings | Google Generative AI (`models/embedding-001`) |
| Language Model | Gemini 1.5 Flash / Pro |
| Vector Store | FAISS |
| RAG Pipeline | LangChain |
| Secrets Management | python-dotenv |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/MultiPDF-Chatbot.git
cd MultiPDF-Chatbot
