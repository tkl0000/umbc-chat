import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain import OpenAI, VectorDBQA
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

#sqlite version fix
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

agent_executor = ""
api_key = ""

# App title
st.set_page_config(page_title="🤗💬 HugChat")

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬 HugChat')
    if ('OPENAI_API_KEY' in st.secrets):
        st.success('OpenAI API Key already provided!', icon='✅')
        openai_key = st.secrets['OPENAI_API_KEY']
        llm = OpenAI(openai_api_key=openai_key, temperature=0)
        loader = TextLoader("./qna.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        qna_store = Chroma.from_documents(
            texts, embeddings, collection_name="qna"
        )
        vectorstore_info = VectorStoreInfo(
            name="QnA",
            description="Frequently asked questions about this semester.",
            vectorstore=qna_store,
        )
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
        agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)
    else:
        api_key = st.text_input('Enter OpenAI API Key:', type='password')
        if (api_key == ""):
            st.warning('Please enter your API Key!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input, agent):
    # # Hugging Face Login
    # sign = Login(email, passwd)
    # cookies = sign.login()
    # # Create ChatBot                        
    # chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return agent.run(prompt_input)

# User-provided prompt
if prompt := st.chat_input(disabled=not (api_key == "")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, agent_executor) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)