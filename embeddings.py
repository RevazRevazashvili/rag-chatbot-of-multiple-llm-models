from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
import os
from langchain_community.vectorstores import Pinecone
import document_processing

os.environ['PINECONE_API_KEY'] = '25607a32-7de3-475e-90c1-49b65378550f'


# function to load pdf document
def load_docs(directory):
    loader = PyPDFLoader(directory)
    documents = loader.load()
    return documents


# load document
documents = load_docs('data/Machine Learning- Step-by-Step Guide To Implement Machine Learning Algorithms with Python ( PDFDrive ).pdf')


# splitting document with chunk_size
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


# docs = split_docs(documents)  # text spliter
docs = document_processing.docs_list


# model to create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# Vector DataBase
# initialize pinecone
environment = "gcp-starter"
pc = pinecone.Pinecone(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=environment
    )

# database index name
index_name = "chat"  # your index name here

# storing embeddings on database
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
