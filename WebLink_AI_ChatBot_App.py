import sys
import os
import asyncio

# ---------------- WINDOWS asyncio fix ---------------- #
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------- IMPORTS ---------------- #
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Helps prevent some websites blocking requests
os.environ["USER_AGENT"] = "Mozilla/5.0"

# ---------------- OPENAI API KEY ---------------- #
os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE"

# ---------------- STREAMLIT UI ---------------- #
st.title("📊 Equity Research AI Agent")
st.write("Ask questions about financial news articles.")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("News Article URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")
process_button = st.sidebar.button("Process Articles")

# ---------------- HELPER FUNCTION ---------------- #
def format_docs_runnable(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in docs
    )

# ---------------- PROCESS DOCUMENTS ---------------- #
if process_button:

    urls = [url for url in [url1, url2, url3] if url != ""]

    if len(urls) == 0:
        st.sidebar.warning("Please enter at least one URL")
        st.stop()

    with st.spinner("Loading and processing articles..."):
        loader = WebBaseLoader(urls)
        data = loader.load()

        # Debug
        with st.expander("Raw loaded data"):
            if len(data) == 0:
                st.warning("No content loaded from the URL.")
            for i, doc in enumerate(data):
                st.write(f"--- Document {i+1} ---")
                st.write(doc.page_content[:500])

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(data)
        st.write("Docs length:", len(docs))

        if len(docs) == 0:
            st.error("No text extracted from the article.")
            st.stop()

        # Embeddings
        embeddings = OpenAIEmbeddings()

        persist_directory = "./chroma_db"

        if not os.path.exists(persist_directory):
        
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name="equity_research_docs"
            )
        
            vectorstore.persist()
        
        else:
        
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name="equity_research_docs"
            )
        st.write("Vector store ready ✅")

        # Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.session_state.retriever = retriever

        st.write("Retriever created ✅")

        # Prompt
        prompt = ChatPromptTemplate.from_template("""
You are a financial research assistant.

Use ONLY the context below to answer.

If the answer is not in the context, say:
"I could not find that information in the provided articles."

Context:
{context}

Question:
{question}

Provide a concise answer with key insights.

Answer:
""")

        # RAG chain
        rag_chain = (
            {
                "context": retriever | format_docs_runnable,
                "question": RunnablePassthrough()
            }
            | prompt
            | ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)
            | StrOutputParser()
        )

        st.session_state.rag_chain = rag_chain

    st.sidebar.success("Articles processed!")

# ---------------- MAIN QUESTION AREA ---------------- #
question = st.text_input("Ask a question about the articles")

if question:

    if "rag_chain" not in st.session_state or "retriever" not in st.session_state:
        st.warning("Please process URLs first using the sidebar.")
    else:

        with st.spinner("Thinking..."):

            retriever = st.session_state.retriever
            rag_chain = st.session_state.rag_chain

            docs = retriever.invoke(question)
            st.write("Docs length:", len(docs))

            response = rag_chain.invoke(question)

        st.subheader("Answer")
        st.write(response)

        # Sources
        st.subheader("Sources")

        sources = set(doc.metadata.get("source") for doc in docs)

        for src in sources:
            st.markdown(f"[{src}]({src})")