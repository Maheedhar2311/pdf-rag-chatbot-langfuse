from rag.pipeline import RAGPipeline


def main():

    pipeline = RAGPipeline(
        pdf_path="data/placement_and_internship_policy.pdf"
    )

    pipeline.build()

    while True:

        query = input("\nAsk Question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        results = pipeline.retrieve(query)

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

            print(docs[i][:500])

            print("\nMetadata")

            print(metas[i])

            print("\nDistance")

            print(distances[i])

            print("=" * 80)


if __name__ == "__main__":
    main()