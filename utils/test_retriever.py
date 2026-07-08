from rag.pdf_loader import PDFLoader
from rag.splitter import TextSplitter
from rag.embeddings import EmbeddingModel
from rag.vectordb import VectorStore
from rag.retriever import Retriever


def main():

    # -----------------------------
    # Load PDF
    # -----------------------------
    loader = PDFLoader(
        "data/placement_and_internship_policy.pdf"
    )

    documents = loader.load()

    # -----------------------------
    # Split into chunks
    # -----------------------------
    splitter = TextSplitter()

    chunks = splitter.split_documents(documents)

    # -----------------------------
    # Load embedding model
    # -----------------------------
    embedding_model = EmbeddingModel()

    # -----------------------------
    # Generate embeddings
    # -----------------------------
    embeddings = embedding_model.embed_documents(
        chunks
    )

    # -----------------------------
    # Create fresh Chroma collection
    # -----------------------------
    vector_store = VectorStore(
        reset=True
    )

    # -----------------------------
    # Store chunks
    # -----------------------------
    vector_store.add_documents(
        chunks,
        embeddings
    )

    # -----------------------------
    # Retriever
    # -----------------------------
    retriever = Retriever(
        embedding_model,
        vector_store
    )

    # -----------------------------
    # Query
    # -----------------------------
    query = input("\nAsk a question: ")

    results = retriever.retrieve(
        query,
        top_k=5
    )

    print("\n")
    print("=" * 80)
    print("Retrieved Chunks")
    print("=" * 80)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(docs)):

        print(f"\nResult {i+1}")

        print("-" * 60)

        print(docs[i])

        print("\nMetadata:")

        print(metas[i])

        print("\nDistance:")

        print(distances[i])

        print("=" * 80)


if __name__ == "__main__":
    main()