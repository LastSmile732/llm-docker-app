import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document

class DocReader:
    def __init__(self, pdf_path, model_path="/opt/all-mpnet-base-v2", persist_directory="db"):
        self.pdfs = glob.glob(f"{pdf_path}/*.pdf")  # Adjusted to get all PDF files in the folder
        self.model_path = model_path
        self.persist_directory = persist_directory

    def load_pdfs(self):
        all_pages = []
        for pdf_file in self.pdfs:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            all_pages.extend(pages)
        return all_pages

    def convert_to_markdown(self, documents):
        markdown_text = ""
        for doc in documents:
    
            page_text = doc.page_content.replace('\n', '\n\n')  # Add extra newline for Markdown
            markdown_text += page_text + "\n\n---\n\n"
        return markdown_text

    def split_text(self, pages):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=128, 
            chunk_overlap=24)
        documents = [Document(page_content=page) for page in pages]
        split_documents = text_splitter.split_documents(documents)
        texts = [doc.page_content for doc in split_documents]

        return texts

    def generate_embeddings(self, texts):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_path,
            model_kwargs={"device": "cuda:0"},
            encode_kwargs={"normalize_embeddings": True},
        )
        documents = [Document(page_content=text) for text in texts]

        db = Qdrant.from_documents(documents, embeddings, location=":memory:", collection_name="pdf_collection")
        return db

    def search_similar(self, input_text, k=3):
        results = self.db.similarity_search(input_text, k)
        return results
