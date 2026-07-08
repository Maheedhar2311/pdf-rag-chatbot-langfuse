import chromadb


class VectorStore:
    """
    Handles storage and retrieval using ChromaDB.
    """

    def __init__(
        self,
        collection_name="placement_policy",
        persist_directory="./chroma_db",
        reset=False
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        if reset:
            try:
                self.client.delete_collection(
                    name=collection_name
                )
                print(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, chunks, embeddings):

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=[doc.page_content for doc in chunks],
            metadatas=[doc.metadata for doc in chunks],
            embeddings=embeddings.tolist()
        )

        print(f"Stored {len(chunks)} chunks successfully.")

    def query(self, query_embedding, top_k=5):

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        return results