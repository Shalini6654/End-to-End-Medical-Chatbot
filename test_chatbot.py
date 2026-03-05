import os
import sys
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_embeddings
from src.prompt import system_prompt

def test_chatbot_accuracy():
    print("Loading environment variables...")
    load_dotenv()
    
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    if not PINECONE_API_KEY or not GROQ_API_KEY:
        print("Error: API Keys are missing. Please ensure .env is correctly configured.")
        return

    print("Initializing embeddings and vector store...")
    embeddings = download_embeddings()
    index_name = "medical-chatbot"
    
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    chatmodel = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    test_questions = [
        "What are the symptoms of acne?",
        "What is paracetamol used for?",
        "Can you explain what a fever is and how to treat it?",
        "What is the capital of France?" # Testing out-of-domain rejection
    ]

    print("\n--- Starting Evaluation ---\n")
    
    for idx, question in enumerate(test_questions, 1):
        print(f"Q{idx}: {question}")
        try:
            # Let's inspect context retrieved
            docs = retriever.invoke(question)
            print(f"-> Retrieved {len(docs)} documents for context.")
            
            response = rag_chain.invoke({"input": question})
            print(f"A{idx}: {response['answer']}\n")
        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            
if __name__ == "__main__":
    test_chatbot_accuracy()
