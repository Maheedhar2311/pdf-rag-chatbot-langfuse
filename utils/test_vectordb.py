from rag.pdf_loader import PDFLoader
from rag.splitter import TextSplitter
from rag.embeddings import EmbeddingModel
from rag.vectordb import VectorStore


def main():

    loader = PDFLoader(
        "data/placement_and_internship_policy.pdf"
    )

    documents = loader.load()

    splitter = TextSplitter()

    chunks = splitter.split_documents(documents)

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(chunks)

    db = VectorStore()

    db.add_documents(chunks, embeddings)

    print("=" * 60)
    print("Documents stored successfully!")
    print("=" * 60)

    query = input("\nEnter your question: ")

    query_embedding = embedding_model.embed_query(query)

    results = db.query(query_embedding)

    print("\n")
    print("=" * 60)
    print("Top Retrieved Chunks")
    print("=" * 60)

    for i, document in enumerate(results["documents"][0], start=1):

        print(f"\nResult {i}")

        print("-" * 40)

        print(document[:600])

        print()

        print(results["metadatas"][0][i - 1])

        print("=" * 60)


if __name__ == "__main__":
    main()