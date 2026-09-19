import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 8

# Current Groq model used for text generation
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv("config.env", override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in config.env"
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    index,
    metadata,
    embedding_model,
    top_k=TOP_K
):
    """
    Convert the question into an embedding and retrieve
    the most relevant chunks from the uploaded documents.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_documents = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        document = metadata[idx].copy()

        document["distance"] = float(distance)

        retrieved_documents.append(document)

    return retrieved_documents


# ============================================================
# CREATE CONTEXT
# ============================================================

def create_context(documents):
    """
    Combine retrieved chunks into a context string.
    """

    context_parts = []

    for i, document in enumerate(
        documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Document: {document['source']}
Page: {document['page']}

Content:
{document['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    documents,
    conversation_history=None
):
    """
    Generate a grounded educational answer using
    ONLY the retrieved document context.

    conversation_history is optional and is used only
    to understand follow-up questions.
    """

    if not documents:
        return (
            "I could not find relevant information "
            "in the uploaded document(s)."
        )

    context = create_context(documents)

    system_prompt = """
You are a Medical Information RAG Assistant.

Your job is to answer GENERAL EDUCATIONAL
medical-information questions using ONLY the
retrieved content from the uploaded document(s).

IMPORTANT RULES:

1. Use ONLY the supplied document context as evidence.
2. Do NOT use outside medical knowledge as evidence.
3. Do NOT invent or guess facts.
4. If the retrieved context does not support the answer,
   say exactly:
   "I could not find this information in the uploaded document(s)."
5. Do NOT diagnose the user.
6. Do NOT provide personalized treatment.
7. Do NOT provide medication dosage or prescriptions.
8. Keep the answer clear and easy to understand.
9. At the end, mention the relevant document name(s)
   and page number(s).
10. Previous conversation is only for understanding the
    user's wording or follow-up question. It is NOT evidence.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add previous conversation only for conversational continuity.
    # It is explicitly NOT treated as evidence.
    if conversation_history:

        for message in conversation_history:

            role = message.get("role")
            content = message.get("content")

            if role in ("user", "assistant") and content:

                messages.append(
                    {
                        "role": role,
                        "content": content
                    }
                )

    user_prompt = f"""
RETRIEVED DOCUMENT CONTEXT
===========================

{context}

===========================
CURRENT USER QUESTION
===========================

{question}

Answer the current question using ONLY the retrieved
document context above.
"""

    messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1
    )

    answer = response.choices[0].message.content

    if not answer:
        return (
            "I could not generate an answer from "
            "the uploaded document(s)."
        )

    return answer


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def ask_question(
    question,
    index,
    metadata,
    embedding_model,
    conversation_history=None
):
    """
    Complete RAG pipeline:

    Question
        ↓
    Query embedding
        ↓
    FAISS retrieval
        ↓
    Relevant chunks
        ↓
    Groq
        ↓
    Grounded answer
    """

    documents = retrieve_documents(
        question,
        index,
        metadata,
        embedding_model,
        TOP_K
    )

    answer = generate_answer(
        question,
        documents,
        conversation_history
    )

    return answer, documents