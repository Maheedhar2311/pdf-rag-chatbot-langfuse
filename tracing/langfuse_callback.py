from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST
)

Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)

langfuse = get_client()

langfuse_handler = CallbackHandler()