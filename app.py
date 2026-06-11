import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Page config
st.set_page_config(
    page_title="LLM-Powered Study Assistant",
    page_icon="📚",
    layout="wide"
)
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
}



.metric-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e6e6e6;
}

.chat-container {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# 📚 AI Study Assistant

Upload one or more PDFs and ask questions using Retrieval-Augmented Generation (RAG).

Powered by FAISS, Sentence Transformers, and Llama 3.
""")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Upload PDF
uploaded_files = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    text = ""

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    st.success("✅ PDF uploaded successfully!")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📚 PDFs", len(uploaded_files))

    with col2:
        st.metric("✂️ Chunks", len(chunks))

    with col3:
        st.metric("📝 Characters", f"{len(text):,}")

    # Sidebar info
    with st.sidebar:

        st.title("📚 Study Assistant")

        st.markdown("---")

        st.subheader("Uploaded PDFs")

        for pdf in uploaded_files:
            st.write(f"📄 {pdf.name}")

        st.markdown("---")

        st.subheader("Statistics")

        st.write(f"📚 PDFs: {len(uploaded_files)}")
        st.write(f"📝 Characters: {len(text):,}")
        st.write(f"✂️ Chunks: {len(chunks)}")

        st.markdown("---")
        # Download Chat History
        if st.session_state.chat_history:

            chat_text = ""

            for item in st.session_state.chat_history:

                chat_text += f"""
        User: {item['question']}
        Assistant: {item['answer']}

        """

            st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name="study_session.txt",
            mime="text/plain"
            )

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Create embeddings and vector store
    with st.spinner("Creating vector database..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embeddings
        )

    st.success("✅ Document ready for questions!")

    # Display previous chat history
    for item in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

    # Chat input
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        # Retrieve relevant chunks
        results = vector_store.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        # Build memory context
        history = ""

        for item in st.session_state.chat_history[-3:]:

            history += f"""
User: {item['question']}
Assistant: {item['answer']}
"""

        prompt = f"""
You are a helpful study assistant.

Conversation History:
{history}

Retrieved Context:
{context}

Question:
{question}

Instructions:
- Answer only using the retrieved context.
- If the answer is not available, respond:
  "I could not find this information in the uploaded document."
- Keep answers concise and accurate.
"""

        # Generate answer
        with st.spinner("Generating answer..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.write(answer)

        # Save history
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        # Show retrieved chunks
        with st.expander("📖 Retrieved Source Chunks"):

            for i, doc in enumerate(results):

                st.markdown(f"### Source Chunk {i+1}")
                st.write(doc.page_content)
                st.divider()