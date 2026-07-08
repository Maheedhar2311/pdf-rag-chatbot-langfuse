def build_prompt(question, retrieved_documents):
    """
    Build a RAG prompt using the retrieved documents.
    """

    context = "\n\n".join(
        [doc for doc in retrieved_documents]
    )

    prompt = f"""
You are an intelligent AI assistant.

Answer the user's question ONLY using the information provided in the context.

Rules:

- Do not use outside knowledge.
- If the answer is not present in the context, say:
  "I couldn't find the answer in the provided document."
- Keep the answer clear and concise.

=========================
Context
=========================

{context}

=========================
Question
=========================

{question}

=========================
Answer
=========================
"""

    return prompt