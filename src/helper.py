from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchai.schema import Document
from typing import List

# 1. Extract Data from the PDF file
def load_pdf_file(data):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# 2. Split documents into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(documents)
    return texts

# 3. Initialize embeddings
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return embeddings

# 4. Initialize Pinecone vector store
def get_vector_store():
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )
    return vectorstore

# 5. Initialize Groq LLM
def get_llm():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.7
    )
    return llm

# 6. Create RAG chain
def get_rag_chain():
    vectorstore = get_vector_store()
    llm = get_llm()
    
    rag_chain = (vectorstore | llm)
    return rag_chain