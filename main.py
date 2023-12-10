import os
from typing import List
import properties
import json
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from httpcore import request
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

load_dotenv()

prompt_template = """

Use the following context to answer the question. 

Say "Sorry I don't know the answer" if you don't know the answer. 

{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(prompt_template)

def get_chunks(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap=50)
    
    chunks = splitter.split_documents(data)

    return chunks

def get_vdb_retriever(chunks):
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks,embeddings)

    return db.as_retriever()

def load_data():
    print("****This was not working if we upload pdf from api/postman was giving pdfminer error so I have attached pdf path and you can replace with your own pdf path")
    pdf_loader = PyPDFLoader("enter your pdf path here")
    data = pdf_loader.load()

    return data

def retrieve_qa(q):
    
    llm = OpenAI()

    data = load_data()

    chunks = get_chunks(data)

    db_retriever = get_vdb_retriever(chunks)

    retrievalQA = RetrievalQA.from_chain_type(llm, 
        chain_type="stuff", 
        retriever=db_retriever,
        chain_type_kwargs={"prompt": prompt})

    return retrievalQA({"query": q})

def ask(question):
    return retrieve_qa(question)

@app.post("/query")
async def query(files: List[UploadFile] = File(...)):
    os.environ["OPENAI_API_KEY"] = properties.APIKEY
    for file in files:
        if file.filename == "inputfile.json":
            json_data = json.load(file.file)
            
    response = dict()
    for i in range(0, len(json_data)):
        query = json_data[i]['content']
        response[i] = ask(query)

    return response
