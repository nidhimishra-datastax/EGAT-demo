# bring in our LLAMA_CLOUD_API_KEY
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.astra_db import AstraDBVectorStore

load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# set up parser
parser = LlamaParse(
    result_type="markdown",
    parsing_instruction = "You are parsing an extract from EGAT's medical document. This extract talks gives an Explanation of procedures for operating according to the QWP for medical treatment reimbursement Government hospital (inside patients/outside patients). Extract all the Steps for carrying out work and their relevant details. Also add a chunk that summarises all the steps",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    vendor_multimodal_api_key="sk-proj-lGcvgmE3P1YBHhHXGasGT3BlbkFJc8t1Zc5r4YnyYKCDXmkZ",
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['egatTH.docx'], file_extractor=file_extractor).load_data()
print(documents)


#load the extracted data in AstraDB
astra_db_store = AstraDBVectorStore(
    token=os.environ.get('ASTRA_TOKEN'),
    api_endpoint=os.environ.get('ASTRA_ENDPOINT'),
    collection_name="EGATTHLLAMA",
    embedding_dimension=1536,
)
# create an index from the parsed markdown
storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# create a query engine for the index
query_engine = index.as_query_engine()

# query the engine
query_engine = index.as_query_engine()
query = "What is the procedure for medical treatment reimbursement "
response = query_engine.query(query)
print(response)