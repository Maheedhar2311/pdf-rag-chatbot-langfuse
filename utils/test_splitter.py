from rag.pdf_loader import PDFLoader
from rag.splitter import TextSplitter


def main():

    loader = PDFLoader("data/placement_and_internship_policy.pdf")
    documents = loader.load()

    splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print("=" * 60)
    print("Total Chunks:", len(chunks))
    print("=" * 60)

    print("\nFirst Chunk Metadata:\n")
    print(chunks[0].metadata)

    print("\nFirst Chunk Preview:\n")
    print(chunks[0].page_content)

    print("\n")

    print("=" * 60)

    print("Second Chunk Preview:\n")

    print(chunks[1].page_content)


if __name__ == "__main__":
    main()