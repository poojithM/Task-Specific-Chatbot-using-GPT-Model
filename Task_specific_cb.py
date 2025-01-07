import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv

load_dotenv()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    
system_msg = SystemMessagePromptTemplate.from_template("You are a stand-up comedian AI assistant. Reply to messages more joyfully!")
human_msg = HumanMessagePromptTemplate.from_template("{input}")

chat_prompt = ChatPromptTemplate.from_messages([
    system_msg,
    MessagesPlaceholder(variable_name="history"),  # Includes history from memory
    human_msg
])




def get_response(question):
    llm = ChatOpenAI(temperature = 0.6, openai_api_key = os.environ["OPENAI_API_KEY"])
    
    
    
    chain = ConversationChain(
        llm = llm,
        memory = st.session_state.memory,
        prompt = chat_prompt,
        verbose = True
    )
    
    return chain.predict(input = question)

st.set_page_config("Chat Bot")

st.header("What Can I Help With?")

input = st.text_input("Message",key = "input")

response = get_response(input)

submit = st.button("Enter")


if submit:
    st.subheader("Response: ")
    st.write(response)