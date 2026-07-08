from rag.llm import GroqLLM


def main():

    llm = GroqLLM()

    answer = llm.generate(
        "Explain what Artificial Intelligence is in one sentence."
    )

    print(answer)


if __name__ == "__main__":
    main()