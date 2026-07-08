from rag.pdf_loader import PDFLoader
from rag.splitter import TextSplitter
from rag.embeddings import EmbeddingModel


def main():

    loader = PDFLoader("data/placement_and_internship_policy.pdf")

    documents = loader.load()

    splitter = TextSplitter()

    chunks = splitter.split_documents(documents)

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(chunks)

    print("\n")

    print("=" * 60)
    print("Total Chunks :", len(chunks))
    print("Embedding Shape :", embeddings.shape)
    print("=" * 60)


if __name__ == "__main__":
    main()