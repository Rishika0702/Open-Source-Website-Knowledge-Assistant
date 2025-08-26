import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

st.title("🌐 Open-Source Website Knowledge Assistant")
st.caption("Ask questions from any open-source website and get context-aware answers instantly.")

url = st.text_input("Enter a website/blog/article URL:", placeholder="https://example.com")

if url:
    with st.spinner("🔍 Fetching and indexing website content..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])

        vectorstore = FAISS.from_documents(final_documents, embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

        prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based only on the provided context.
        Be clear, concise, and helpful.

        <context>
        {context}
        </context>

        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        st.success("✅ Website fetched successfully! Now ask your questions below.")

        user_query = st.text_input("💬 Ask a question about the website content:")

        if user_query:
            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_query})
            elapsed = round(time.process_time()-start, 2)

            st.markdown(
                f"""
                <div style="padding:15px; border-radius:12px; background-color:#f0f2f6; border:1px solid #ddd;">
                <h4>🤖 Answer</h4>
                <p style="font-size:16px;">{response['answer']}</p>
                <small>⏱️ Response time: {elapsed}s</small>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Show retrieved context
            with st.expander("📑 Relevant Document Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(
                        f"""
                        <div style="padding:10px; margin-bottom:8px; border-radius:10px; background-color:#ffffff; border:1px solid #ddd;">
                        <b>Chunk {i+1}</b><br>
                        {doc.page_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
