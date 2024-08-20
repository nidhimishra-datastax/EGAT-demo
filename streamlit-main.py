import streamlit as st
import os
import logging
import sys
from dotenv import load_dotenv
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import PromptTemplate
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from IPython.display import Markdown, display

st.set_page_config(page_title="EGAT HR Assistant", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("EGAT HR Assistant")
st.image("EGATlogo.png", width=None)
st.header("Chat with the HR Medical Policy Document", divider="blue")

load_dotenv()

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "EGAT Virtual Assistant welcomes you!",
        }
    ]

# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}" f"**Text:** "
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown(""))

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    #load the extracted data in AstraDB
    astra_db_store = AstraDBVectorStore(
        token=os.environ.get('ASTRA_TOKEN'),
        api_endpoint=os.environ.get('ASTRA_ENDPOINT'),
        collection_name="EGATTHLLAMA",
        embedding_dimension=1536,
    )
    new_index_instance = VectorStoreIndex.from_vector_store(
        vector_store=astra_db_store
    )
    return new_index_instance

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    print("Initialising the engine")
    index = load_vectorstore()

    # now you can do querying, etc:
    query_engine = index.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")

    #Accessing Prompts
    prompts_dict = query_engine.get_prompts()
    display_prompt_dict(prompts_dict)

    #Customising Prompts
    new_summary_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "You are EGAT's HR Assistant chatbot. Given the context information and not prior knowledge, "
        "Answer the query in professional and descriptive tone. \n"
        "Answer in Thai only. \n"
        "Query: {query_str}\n"
        "Answer: "
    )
    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:summary_template": new_summary_tmpl}
    )

    #Accessing Prompts
    prompts_dict = query_engine.get_prompts()
    display_prompt_dict(prompts_dict)
    print("Prompt updated")


if prompt := st.chat_input(
    "How can I help?"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        st.spinner("Thinking...")
        print("Query box")
        
        #Accessing Prompts
        prompts_dict = query_engine.get_prompts()
        display_prompt_dict(prompts_dict)

        #response = query_engine.query(prompt)
        response = query_engine.query(prompt)
        print(response)
        st.write(response.response)
        #st.write_stream(response_stream)
        message = {"role": "assistant", "content": response.response}
        # Add response to message history
        st.session_state.messages.append(message)
