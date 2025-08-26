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

if "chat_ready" not in st.session_state:
    st.session_state.chat_ready = False
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.chat_ready:
    st.caption("Enter a website/blog/article URL and start chatting with it.")

    url = st.text_input("Enter a website/blog/article URL:", placeholder="https://example.com")

    if url:
        with st.spinner("🔍 Fetching website content..."):
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

            st.session_state.retrieval_chain = retrieval_chain
            st.session_state.chat_ready = True

            st.success("✅ Website fetched successfully! Redirecting to chat...")

            st.experimental_rerun()

else:
    st.caption("✅ Website fetched! Now chat with the bot below.")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("💬 Ask me anything about this website..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        start = time.process_time()
        response = st.session_state.retrieval_chain.invoke({"input": query})
        elapsed = round(time.process_time() - start, 2)

        answer = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(f"{answer}\n\n ⏱️ *Response time: {elapsed}s*")

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
