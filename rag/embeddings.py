from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Generates embeddings for documents and queries.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        print("=" * 60)
        print("Loading Embedding Model...")
        print("=" * 60)

        self.model = SentenceTransformer(model_name)

        print("Embedding Model Loaded Successfully!")

    def embed_documents(self, documents):
        """
        Generate embeddings for LangChain Documents.
        """

        texts = [doc.page_content for doc in documents]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings

    def embed_query(self, query):
        """
        Generate embedding for a user query.
        """

        return self.model.encode(
            query,
            convert_to_numpy=True
        )