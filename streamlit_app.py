import streamlit as st
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="Medicine Chatbot", page_icon="🤖")

# Inject Custom CSS to keep the same gradient background design pattern
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, rgb(38, 51, 61), rgb(50, 55, 65), rgb(33, 33, 78));
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header layout resembling the original HTML
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387569.png", width=60)
with col2:
    st.title("Medical Chatbot")
    st.write("Ask me anything!")

st.divider()

# Initialize LangChain components (Cached to avoid re-running every interaction)
@st.cache_resource
def init_rag_chain():
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    
    embeddings = download_embeddings()
    index_name = "medical-chatbot"
    
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    chatmodel = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

rag_chain = init_rag_chain()

# Avatar URLs matching your original HTML structure
user_avatar = "https://i.ibb.co/d5b84Xw/Untitled-design.png"
bot_avatar = "https://cdn-icons-png.flaticon.com/512/387/387569.png"

# Setup Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    avatar = user_avatar if message["role"] == "user" else bot_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input & Form Processing
if prompt := st.chat_input("Type your message..."):
    # Display User Message
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get Chatbot Response
    with st.spinner("Typing..."):
        try:
            response = rag_chain.invoke({"input": prompt})
            bot_response = response["answer"]
        except Exception as e:
            bot_response = f"An error occurred: {str(e)}"
    
    # Display Bot Message
    with st.chat_message("assistant", avatar=bot_avatar):
        st.markdown(bot_response)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
