from pypdf import PdfReader
from langchain_core.documents import Document


class PDFLoader:
    """
    Custom PDF loader using pypdf.
    Returns a list of LangChain Document objects.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self):
        documents = []

        reader = PdfReader(self.pdf_path)

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()

            # Handle pages with no extractable text
            if text is None:
                text = ""

            document = Document(
                page_content=text,
                metadata={
                    "source": self.pdf_path,
                    "page": page_number + 1
                }
            )

            documents.append(document)

        return documents