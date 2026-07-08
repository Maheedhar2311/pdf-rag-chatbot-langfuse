from rag.pdf_loader import PDFLoader
from rag.splitter import TextSplitter
from rag.embeddings import EmbeddingModel
from rag.vectordb import VectorStore
from rag.retriever import Retriever
from rag.llm import GroqLLM
from rag.prompt import build_prompt
from tracing.langfuse_callback import langfuse
from langfuse import propagate_attributes

class RAGPipeline:

    def __init__(
        self,
        pdf_path,
        chunk_size=1000,
        chunk_overlap=200
    ):

        self.pdf_path = pdf_path

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore(
            reset=True
        )

        self.retriever = Retriever(
            self.embedding_model,
            self.vector_store
        )

        self.llm = GroqLLM()

    def build(self):

        print("=" * 60)
        print("Building RAG Pipeline")
        print("=" * 60)

        loader = PDFLoader(self.pdf_path)

        documents = loader.load()

        splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = splitter.split_documents(documents)

        embeddings = self.embedding_model.embed_documents(
            chunks
        )

        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        print("\nPipeline Built Successfully!")

        print(f"Pages Loaded : {len(documents)}")

        print(f"Chunks Created : {len(chunks)}")

    def retrieve(
        self,
        question,
        top_k=5
    ):

        return self.retriever.retrieve(
            question,
            top_k
        )

    def ask(
    self,
    question,
    top_k=5
    ):

        with langfuse.start_as_current_observation(
            as_type="span",
            name="PDF RAG Query",
            input={
                "question": question
            }
        ) as root:

            results = self.retrieve(
                question,
                top_k
            )

            retrieved_docs = results["documents"][0]

            metadata = results["metadatas"][0]

            distances = results["distances"][0]

            context = "\n\n".join(retrieved_docs)

            with root.start_as_current_observation(
                as_type="retriever",
                name="Vector Retrieval"
            ) as retrieval:

                retrieval.update(
                    input={
                        "question": question
                    },
                    output={
                        "context": context,
                        "chunks": retrieved_docs,
                        "metadata": metadata,
                        "distances": distances
                    }
                )

            prompt = build_prompt(
                question,
                retrieved_docs
            )

            with propagate_attributes(
                metadata={
                    "question": question,
                    "retrieved_chunks": len(retrieved_docs),
                    "source": self.pdf_path
                }
            ):

                answer = self.llm.generate(prompt)

            root.update(
                output={
                    "answer": answer
                }
            )

            langfuse.flush()

            return {
                "question": question,
                "context": context,
                "prompt": prompt,
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "metadata": metadata,
                "distances": distances
            }