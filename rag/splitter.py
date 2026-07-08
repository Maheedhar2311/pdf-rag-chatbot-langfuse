from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """
    Splits LangChain Documents into overlapping chunks.
    """

    def __init__(self,
                 chunk_size=1000,
                 chunk_overlap=200):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)