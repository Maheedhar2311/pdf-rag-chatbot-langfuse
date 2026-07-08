from rag.pipeline import RAGPipeline


def main():

    pipeline = RAGPipeline(
        pdf_path="data/placement_and_internship_policy.pdf"
    )

    pipeline.build()

    while True:

        question = input("\nAsk Question (exit to quit): ")

        if question.lower() == "exit":
            break

        response = pipeline.ask(question)

        print()

        print("=" * 80)

        print("Answer")

        print("=" * 80)

        print(response["answer"])


if __name__ == "__main__":
    main()