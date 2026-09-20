import os

import numpy as np
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 8

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
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(query_vector, document_vectors):
    """
    Calculate cosine similarity between one query vector
    and all document vectors.
    """

    query_norm = np.linalg.norm(query_vector)

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    # Prevent division by zero
    query_norm = max(query_norm, 1e-12)

    document_norms = np.maximum(
        document_norms,
        1e-12
    )

    similarities = (
        document_vectors @ query_vector
    ) / (
        document_norms * query_norm
    )

    return similarities


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    embedding_matrix,
    metadata,
    embedding_model,
    top_k=TOP_K
):
    """
    Retrieve the most relevant chunks using NumPy
    cosine similarity.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )[0]

    similarities = cosine_similarity(
        question_embedding,
        embedding_matrix
    )

    top_k = min(
        top_k,
        len(similarities)
    )

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    retrieved_documents = []

    for idx in top_indices:

        document = metadata[int(idx)].copy()

        # Store similarity instead of FAISS distance
        document["similarity"] = float(
            similarities[idx]
        )

        retrieved_documents.append(
            document
        )

    return retrieved_documents


# ============================================================
# CREATE CONTEXT
# ============================================================

def create_context(documents):

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
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(
    question,
    documents,
    conversation_history=None
):

    if not documents:

        return (
            "I could not find this information "
            "in the uploaded document(s)."
        )

    context = create_context(
        documents
    )

    system_prompt = """
You are a Medical Information RAG Assistant.

Your job is to answer GENERAL EDUCATIONAL
medical-information questions using ONLY the
retrieved content from the uploaded document(s).

IMPORTANT RULES:

1. Use ONLY the supplied document context.
2. Do NOT use outside medical knowledge as evidence.
3. Do NOT invent or guess facts.
4. If the answer is not supported by the context,
   say exactly:

   "I could not find this information in the
   uploaded document(s)."

5. Do NOT diagnose the user.
6. Do NOT provide personalized treatment.
7. Do NOT provide medication dosage or prescriptions.
8. Keep answers clear and easy to understand.
9. Mention relevant document names and page numbers
   when supported by the context.
10. Conversation history is only for understanding
    follow-up wording. It is NOT evidence.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

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
==========================

{context}

==========================
CURRENT USER QUESTION
==========================

{question}

Answer the current question using ONLY the
retrieved document context above.
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
    embedding_matrix,
    metadata,
    embedding_model,
    conversation_history=None
):

    documents = retrieve_documents(
        question,
        embedding_matrix,
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