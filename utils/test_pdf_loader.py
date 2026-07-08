from rag.pdf_loader import PDFLoader


def main():
    loader = PDFLoader("data/placement_and_internship_policy.pdf")

    documents = loader.load()

    print("=" * 60)
    print(f"Total Pages Loaded: {len(documents)}")
    print("=" * 60)

    print("\nMetadata:")
    print(documents[0].metadata)

    print("\nContent Preview:\n")
    print(documents[0].page_content[:1000])


if __name__ == "__main__":
    main()