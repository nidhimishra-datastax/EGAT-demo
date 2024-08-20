import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate
from IPython.display import Markdown, display

load_dotenv()

# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}" f"**Text:** "
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown(""))

#load the extracted data in AstraDB
astra_db_store = AstraDBVectorStore(
    token=os.environ.get('ASTRA_TOKEN'),
    api_endpoint=os.environ.get('ASTRA_ENDPOINT'),
    collection_name="EGATTHLLAMA",
    embedding_dimension=1536,
)

# Configure LLM 
llmgpt4 = OpenAI(model="gpt-4",temperature=0)

#Build Index
new_index_instance = VectorStoreIndex.from_vector_store(
    vector_store=astra_db_store
)

#Get Query Engine
query_engine = new_index_instance.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")

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

response = query_engine.query(
    "Who are you?"
)
print(response)

""" 

# Configure LLM 
llm = OpenAI(model="gpt-4",temperature=0)

query_engine = index.as_query_engine(
  streaming=True, 
  similarity_top_k=3, 
  llm=llm
)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

query = "What is the procedure for medical treatment reimbursement "
response = query_engine.query(query)
print(response) """