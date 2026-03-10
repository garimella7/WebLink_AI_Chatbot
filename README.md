# Web Content Insight Chatbot (RAG)

This project is an AI-powered **Web Content Insight Chatbot** built using **LangChain, Streamlit, OpenAI, and Chroma Vector Database**.

The chatbot allows users to **paste any webpage URLs**, processes the content, and enables users to **ask questions or extract insights from the provided articles**.

The system uses a **Retrieval-Augmented Generation (RAG)** architecture to retrieve relevant document chunks and generate accurate answers using an LLM.

---

## Features

- Accepts **any webpage URLs** from users
- Automatically loads webpage content
- Splits text into manageable chunks
- Converts text into embeddings
- Stores embeddings in **Chroma Vector Database**
- Uses **semantic search retrieval**
- Generates answers using **GPT-4o-mini**
- Interactive UI built with **Streamlit**

---

## System Architecture

User URLs  
↓  
WebBaseLoader  
↓  
Text Splitter  
↓  
Embeddings  
↓  
Chroma Vector Database  
↓  
Retriever  
↓  
Prompt Template  
↓  
GPT-4o-mini  
↓  
Answer
