from langchain_groq import ChatGroq

from config import GROQ_API_KEY

from tracing.langfuse_callback import (
    langfuse_handler,
    langfuse
)


class GroqLLM:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=GROQ_API_KEY,
            temperature=0
        )

    def generate(
        self,
        prompt,
        metadata=None
    ):

        response = self.llm.invoke(
            prompt,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_tags": ["pdf-rag"],
                    **(metadata or {})
                }
            }
        )

        # Push trace immediately
        langfuse.flush()

        return response.content