class Retriever:
    """
    Retrieves the most relevant chunks from ChromaDB.
    """

    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):

        query_embedding = self.embedding_model.embed_query(
            query
        )

        results = self.vector_store.query(
            query_embedding,
            top_k
        )

        return results