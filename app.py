import langchain
from langchain.retrievers import AmazonKendraRetriever
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader

from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

from dotenv import load_dotenv
import streamlit as st
import os
import boto3
import csv

aws_region = 'us-east-1'

langchain.debug = True
langchain.verbose = True


bedrock_client = boto3.client("bedrock-runtime", aws_region)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-g1-text-02", client=bedrock_client)


def load_documents():
    files = ["3de1d40.txt", "3g3uxrh.txt"]
    return [TextLoader(f"./data/{file}", encoding='utf8') for file in files]

def load_directory():
    dir_loader = DirectoryLoader("./data")
    return [dir_loader]

def load_csv():
    csv_loader = CSVLoader(file_path='./ame-data-oct5.csv')
    return [csv_loader]

def get_faiss_doc_retriver():
    loaders = load_documents()
    index = VectorstoreIndexCreator(vectorstore_cls = FAISS, embedding = bedrock_embeddings).from_loaders(loaders)
    print("loaded indexes")
    return index.vectorstore.as_retriever()


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

retriever = get_faiss_doc_retriver() 

if st.button("Search"):
    with st.spinner("Building response..."):
        model_id = MODEL_MAP.get(llm_model)
        llm = Bedrock(model_id=model_id, region_name=aws_region, 
                        client=bedrock_client, 
                        model_kwargs={ "temperature": temperature})

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa(query)
        st.markdown("### Answer:")
        st.write(response['result'])
