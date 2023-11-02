
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import langchain
from langchain.storage import LocalFileStore
from langchain.retrievers import AmazonKendraRetriever
from langchain.embeddings import BedrockEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.callbacks import StreamlitCallbackHandler

from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.cache import GPTCache, SQLiteCache
from langchain.globals import set_llm_cache

from dotenv import load_dotenv
import streamlit as st
import os
import boto3
import csv
import pandas as pd

aws_region = 'us-east-1'

from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.manager import manager_factory
from gptcache.embedding.langchain import LangChain
import hashlib


# langchain.debug = True
# langchain.verbose = True

persist_directory = './chromadb'

bedrock_client = boto3.client("bedrock-runtime", aws_region)
livestreams = pd.read_csv("ame-data-oct5.csv", usecols=["Task ID","Task Name","EP_NUM (number)","GUEST_1_NAME (short text)","STREAMYARD_URL (url)","YOUTUBE_URL (url)","LINKEDIN_URL (url)","RECORDED_S3_URI (url)","AUDIO_URL (url)"])


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    langchain_embedding = LangChain(embeddings= get_embeddings())
    hashed_llm = get_hashed_name(llm)
    init_similar_cache(cache_obj=cache_obj, data_manager=manager_factory(
                        manager="sqlite,Chromadb",
                        data_dir=f"similar_cache_{hashed_llm}"
                    ),
                embedding=langchain_embedding)

def get_embeddings():
    store = LocalFileStore('./cache')
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-g1-text-02", client=bedrock_client)
    return CacheBackedEmbeddings.from_bytes_store(
                    bedrock_embeddings, store, namespace="bedrock")

def load_directory():
    dir_loader = DirectoryLoader("./data")
    return [dir_loader]

def get_doc_retriver(insert_documents=True):
    embedding = get_embeddings()
    if insert_documents:
        loaders = load_directory()
        index = VectorstoreIndexCreator(vectorstore_cls = Chroma, 
                                    embedding = embedding,
                                    vectorstore_kwargs={'persist_directory': persist_directory}
                ).from_loaders(loaders)
        print("All transcripts are indexed.")
        return index.vectorstore.as_retriever()
    else:
        vectorstore = Chroma(embedding_function= embedding, persist_directory=persist_directory
        )
        print("Loaded indexes from file")
        return vectorstore.as_retriever()

if 'cache' not in st.session_state:
    set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    # set_llm_cache(GPTCache(init_gptcache))
    st.session_state['cache'] = True

if 'retriever' not in st.session_state:
    st.session_state['retriever'] = get_doc_retriver(insert_documents=False) 
retriever = st.session_state['retriever']

st.title("Qna with Rahul")
query = st.text_input("What would you like to know?")
max_tokens = st.number_input('Max Tokens', value=1000)
temperature= st.number_input(label="Temperature",step=.1,format="%.2f", value=0.7)
llm_model = st.selectbox("Select LLM", ["Anthropic Claude V2", 
                                        "Amazon Titan Text Express v1",
                                            "Ai21 Labs Jurassic-2 Ultra"])

MODEL_MAP = {
    "Anthropic Claude V2" : "anthropic.claude-v2",
    "Amazon Titan Text Express v1": "amazon.titan-text-express-v1",
    "Ai21 Labs Jurassic-2 Ultra": "ai21.j2-ultra-v1"
}


if st.button("Search"):
    model_id = MODEL_MAP.get(llm_model)
    st.markdown("### Answer:")
    chat_box = st.empty()
    stream_handler = StreamlitCallbackHandler(chat_box)
    llm = Bedrock(model_id=model_id, region_name=aws_region, 
                    client=bedrock_client, 
                    model_kwargs={ "temperature": temperature},
                    streaming=True,
                    callbacks=[stream_handler],
                    cache=True
                    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    response = qa(query)
    chat_box.write(response['result'])
    st.markdown("### Source Transcript:")
    for document in response['source_documents']:
        task_id = document.metadata['source'].replace("data/", "").replace(".txt", "")
        task = livestreams[livestreams["Task ID"] == task_id]
        st.markdown(f"From `{task.loc[task.index[0], 'Task Name']}`. Watch Complete [Livestream]({task.loc[task.index[0], 'LINKEDIN_URL (url)']})")
        st.write(document.page_content)
