import hashlib
import tempfile

import streamlit as st

from rag.pipeline import RAGPipeline


# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")


# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.header("📄 Document")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

    st.divider()

    if st.session_state.pdf_name:

        st.success(
            f"Loaded PDF:\n\n{st.session_state.pdf_name}"
        )


# ----------------------------------------------------
# Build Pipeline if New PDF Uploaded
# ----------------------------------------------------

if uploaded_pdf is not None:

    pdf_bytes = uploaded_pdf.read()

    current_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    if current_hash != st.session_state.pdf_hash:

        st.session_state.pdf_hash = current_hash

        st.session_state.pdf_name = uploaded_pdf.name

        st.session_state.chat_history = []

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(pdf_bytes)

            pdf_path = tmp.name

        with st.spinner(
            "Indexing PDF and creating embeddings..."
        ):

            pipeline = RAGPipeline(pdf_path)

            pipeline.build()

            st.session_state.pipeline = pipeline

        st.success("Knowledge Base Built Successfully!")


# ----------------------------------------------------
# Display Chat History
# ----------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------

if st.session_state.pipeline is not None:

    question = st.chat_input(
        "Ask anything from the uploaded PDF..."
    )

    if question:

        # -----------------------------
        # Show User Message
        # -----------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # -----------------------------
        # Generate Answer
        # -----------------------------

        with st.spinner(
            "Retrieving relevant context and generating answer..."
        ):

            response = st.session_state.pipeline.ask(
                question
            )

        answer = response["answer"]

        # -----------------------------
        # Show Assistant Message
        # -----------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

else:

    st.info(
        "👈 Please upload a PDF from the sidebar to start chatting."
    )