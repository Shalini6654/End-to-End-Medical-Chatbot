from flask import Flask, render_template, request, jsonify
from src.helper import download_embeddings, split_text, load_pdf_file
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"
# Embed each chunk and upset the embeddings into your Pinecone index.
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

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)