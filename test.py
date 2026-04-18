import streamlit as st
import tempfile
import os
import hashlib
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import uuid
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("📄 RAG Agent (Prod Style)")

# -----------------------------
# Session state
# -----------------------------
if "db" not in st.session_state:
    st.session_state.db = None
if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None
if "threads" not in st.session_state:
    st.session_state.threads=[]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"
    st.session_state.threads.append("thread-1")

if st.sidebar.button("New Chat"):
    new_thread=str(uuid.uuid4())
    st.session_state.thread_id = new_thread
    st.session_state.threads.append(new_thread)

st.sidebar.title("Chats")
for tid in st.session_state.threads:
    if st.sidebar.button(tid[:8],key = tid):
        st.session_state.thread_id = tid
def get_file_hash(file_bytes:bytes)-> str:
    return hashlib.sha256(file_bytes).hexdigest()
# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_bytes=uploaded_file.getvalue()
    current_file_hash=get_file_hash(file_bytes)
    if st.session_state.processed_file_hash != current_file_hash:

        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)  
            docs = loader.load()  

            splitter = RecursiveCharacterTextSplitter(  
            chunk_size=500,  
            chunk_overlap=50  
            )  
            chunks = splitter.split_documents(docs)  

            embeddings = HuggingFaceEmbeddings(  
                model_name="all-MiniLM-L6-v2"  
            )  

            db = Chroma.from_documents(chunks, embeddings)  

            st.session_state.db = db 
            st.session_state.processed_file_hash=current_file_hash
            os.remove(tmp_path)  

        st.success("PDF processed!")  
# -----------------------------
# Prompt (IMPORTANT)
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
("system", """You are a helpful AI assistant.
Answer ONLY using the provided context.
If answer is not in context, say "I don't know".
"""),

MessagesPlaceholder(variable_name="messages"),  

("human", """  
Context:
{context}

Question:
{question}
""")
])

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
model_name="llama-3.1-8b-instant",
temperature=0,
streaming=True
)

# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -----------------------------
# Node
# -----------------------------
def chatbot_node(state: AgentState):
    last_user_message = state["messages"][-1].content

    if st.session_state.db is None:  
        context = "No document uploaded"  
    else:  
        retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})  
        docs = retriever.invoke(last_user_message)  
        context = "\n\n".join([d.page_content for d in docs])  

    chain = prompt | llm  

    response = chain.invoke({  
        "messages": state["messages"],  
        "context": context,  
        "question": last_user_message  
    })  

    return {"messages": [response]}  
# -----------------------------
# Graph
# -----------------------------
@st.cache_resource
def build_app():

    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot_node)
    builder.set_entry_point("chatbot")
    builder.add_edge("chatbot", END)

    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)

app = build_app()

# -----------------------------
# Chat UI
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    app.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config={"configurable": {"thread_id": st.session_state.thread_id}}
    )

# -----------------------------
# Display messages
# -----------------------------
state = app.get_state(
config={"configurable": {"thread_id": st.session_state.thread_id}}
)

messages = state.values.get("messages", []) if state else []
print("msg",messages,st.session_state.thread_id,state)
for msg in messages:
    if isinstance(msg, SystemMessage):
        continue

    role = "user" if isinstance(msg, HumanMessage) else "assistant"  

    with st.chat_message(role):  
        st.markdown(msg.content)
