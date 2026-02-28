from dotenv import load_dotenv
import os 
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print("Loading PDF files...")
extracted_data = load_pdf_file(data="data/")

filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting documents into chunks...")
texts_chunk = text_split(extracted_data)

embeddings = download_embeddings()
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)
