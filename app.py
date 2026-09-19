import streamlit as st

from document_processor import create_vectorstore
from rag_system import ask_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Information RAG Assistant",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Medical Information RAG Assistant")

st.write(
    "Upload one or more medical PDF documents and ask "
    "multiple questions based only on the uploaded content."
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "Educational information only. "
    "This assistant does not diagnose conditions "
    "or provide personalized treatment."
)


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

st.subheader("1. Upload Medical Documents")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    selected_names = [
        file.name for file in uploaded_files
    ]

    st.write("Selected documents:")

    for name in selected_names:
        st.write(f"• {name}")

    if st.button("Process Documents"):

        with st.spinner(
            "Extracting text, creating chunks, "
            "generating embeddings, and building "
            "the knowledge base..."
        ):

            try:

                index, metadata, embedding_model = (
                    create_vectorstore(uploaded_files)
                )

                st.session_state.vectorstore = (
                    index,
                    metadata,
                    embedding_model
                )

                st.session_state.processed_files = (
                    selected_names
                )

                # Clear old conversation whenever
                # a new document set is processed.
                st.session_state.messages = []

                st.success(
                    "Documents processed successfully!"
                )

                st.info(
                    f"Created {len(metadata)} text chunks."
                )

            except Exception as error:

                st.error(
                    f"Error while processing documents: {error}"
                )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

if st.session_state.vectorstore is not None:

    st.subheader("2. Knowledge Base")

    st.success(
        "✅ Uploaded documents are ready for questions."
    )

    st.write("Processed documents:")

    for name in st.session_state.processed_files:
        st.write(f"• {name}")


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.messages:

    st.subheader("3. Conversation")

    for message in st.session_state.messages:

        role = message["role"]
        content = message["content"]

        with st.chat_message(role):

            st.write(content)

            # Show sources only for assistant messages
            if role == "assistant":

                sources = message.get("sources", [])

                if sources:

                    st.markdown("**Sources**")

                    for source in sources:
                        st.write(f"• {source}")


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("4. Ask a Question")

question = st.chat_input(
    "Ask a question about the uploaded document..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Check whether documents have been processed
    # --------------------------------------------------------

    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload and process a medical PDF first."
        )

    else:

        index, metadata, embedding_model = (
            st.session_state.vectorstore
        )

        # ----------------------------------------------------
        # Keep previous conversation for follow-up questions
        # ----------------------------------------------------

        conversation_history = (
            st.session_state.messages.copy()
        )

        # Add current user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Show current user question immediately
        with st.chat_message("user"):
            st.write(question)

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the uploaded documents..."
            ):

                try:

                    answer, documents = ask_question(
                        question,
                        index,
                        metadata,
                        embedding_model,
                        conversation_history
                    )

                except Exception as error:

                    st.error(
                        f"Error while generating the answer: {error}"
                    )

                    st.stop()

            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            st.write(answer)

            # ------------------------------------------------
            # Prepare sources
            # ------------------------------------------------

            not_found_message = (
                "I could not find this information "
                "in the uploaded document(s)."
            )

            sources = []

            # Only show sources when the answer is actually
            # supported by retrieved content.
            if (
                documents
                and not_found_message.lower()
                not in answer.lower()
            ):

                seen_sources = set()

                for document in documents:

                    source = (
                        f"{document['source']} "
                        f"— Page {document['page']}"
                    )

                    if source not in seen_sources:

                        sources.append(source)
                        seen_sources.add(source)

                if sources:

                    st.markdown("**Sources**")

                    for source in sources:
                        st.write(f"• {source}")

        # ----------------------------------------------------
        # Save assistant response + sources
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "This assistant provides general educational "
    "information from user-uploaded documents. "
    "It is not a substitute for professional medical advice."
)