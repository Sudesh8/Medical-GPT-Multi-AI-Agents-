from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


path = "temp_dataset"

all_docs = []  # Store all document chunks


splitter = RecursiveCharacterTextSplitter(
    chunk_size=550, chunk_overlap=150, length_function=len
)
for file in tqdm(os.listdir(path)):
    if file.endswith(".pdf"):  # Process only PDF files
        file_path = os.path.join(path, file)
        loader = PyPDFLoader(file_path)

        # Load raw documents
        raw_docs = loader.load()

        DOCS = splitter.split_documents(raw_docs)
        all_docs.extend(DOCS)


DB = FAISS.from_documents(all_docs, embed_model)

DATABASE_PATH = "DB"

DB.save_local(DATABASE_PATH)

print("DTABASE Saved ........")
