from langchain_community.document_loaders import PyPDFLoader
import getpass
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema import Document
import markdown
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import time
import re
import os
import openai
import json
import streamlit as st
import requests
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
import time
#from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain.output_parsers import PydanticOutputParser


os.environ["GROQ_API_KEY"] = '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'
os.environ["NVIDIA_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = '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'


#--------------------------------------------------------------------------------------
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.model.encode(d,batch_size=64).tolist() for d in documents]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()

embedding_model = CustomEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#--------------------------------------------------------------------------------------
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def pdf_rag(file_path, user_input):

    if file_path is not None:
        with open("temp_file", "wb") as f:
            f.write(file_path.getbuffer())

        
    loader = PyMuPDFLoader("temp_file")
    document = loader.load()

    #--------------------------------------------------------------------------------------
    #loader = PyMuPDFLoader("temp_file")
    #documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    splits = text_splitter.split_documents(documents=document)
    print("************************** check **************************")

#--------------------------------------------------------------------------------------
# Create a LangChain embedding

    from langchain_chroma import Chroma
    vectorstore = Chroma.from_documents(splits, embedding=embedding_model, persist_directory="chroma_langchain_db")
    retriever = vectorstore.as_retriever()
    print("***************************************************************")

    system_prompt = (
        "Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'."
        "Use three sentences maximum and answer in regress manner."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    results = rag_chain.invoke({"input": user_input})

    return results['answer']