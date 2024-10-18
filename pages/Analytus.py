__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import streamlit as st
import pandas as pd
import json

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
import os
import sys
import time
from langchain.chains import create_history_aware_retriever


ICON_BLUE = "transperent_logo.png"

st.logo(ICON_BLUE, icon_image=ICON_BLUE)

# Set the page configuration
st.set_page_config(
    page_title="Lervis Enterprise",
    layout="wide",
    page_icon = ICON_BLUE
) 

openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
groq = st.secrets.db_credentials.groq
nvidia = st.secrets.db_credentials.nvidia

#--------------------------------------------------------------------------------------
from langchain.embeddings import HuggingFaceEmbeddings

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


from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm_nvidia = ChatNVIDIA(model="meta/llama3-70b-instruct",api_key = nvidia)
#--------------------------------------------------------------------------------------

col1, col2, col3 = st.columns([1,4,1])

with col1:
        st.image('transperent_logo.png', width=200)

with col3:
    with st.popover("Usage"):
        st.markdown("""
	        <div style="padding: 10px; font-family: Arial, sans-serif;">
	            <h3 style="text-align: center;">About Prep W Lervis</h3>
	            <p>This agent helps individuals prepare for interviews by delivering pertinent information tailored to their inputs.</p>
	            <h4 Bot Usage Instructions:</h4>
	            <p>This agent's purpose is to:</p>
	            <ul style="margin-left: 20px;">
                	<li>Assist you with educational content related to interviews.</li>
                	<li>Answer your questions specifically about interview preparation topics.</li>
                	<li>Provide structured responses with relevant links and resources.</li>
                	<li>Offer soft skill training materials for interview readiness.</li>
                	<li>Implement RAG (Retrieve and Generate) functionality based on selected dropdown options if the RAG checkbox is checked.</li>
            	    </ul>
	            <h4>RAG Implementation:</h4>
	            <p>If you check the RAG checkbox, you will be able to implement RAG based on selected dropdown options.</p>
	        </div>
        """, unsafe_allow_html=True)

with col2:
    st.header("Chat with your documents")
    data = st.file_uploader("Upload a file",type=['pdf','doc','ppt','pptx','xls','xlsx'])

with col2:
    user_input = st.text_area("Enter your query:",args=(True,))

    generate = st.button("Answer")


if user_input and  generate:
    with col2:
        print(data.type)
        try: 
            if  data.type == 'application/pdf':
                from langchain.document_loaders import PyMuPDFLoader
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                if data is not None:
                    with open("temp_file", "wb") as f:
                        f.write(data.getbuffer())

                
                loader = PyMuPDFLoader("temp_file")
                document = loader.load()

                #--------------------------------------------------------------------------------------
                #loader = PyMuPDFLoader("temp_file")
                #documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
                splits = text_splitter.split_documents(documents=document)
                print("************************** splits check **************************")
                #print(splits)
                print("************************** check **************************")

            #--------------------------------------------------------------------------------------
            # Create a LangChain embedding

                from langchain_chroma import Chroma
                import chromadb
                persistent_client = chromadb.PersistentClient()
                persistent_client.delete_collection("langchain")
                collection = persistent_client.get_or_create_collection("langchain")
                #collection.delete()

                vectorstore = Chroma(
                    client=persistent_client,
                    collection_name="langchain",
                    embedding_function=embedding_model,
                )
                vectorstore.add_documents(splits)
                #vectorstore = Chroma.from_documents(splits, embedding=embedding_model, persist_directory="./chroma_langchain_db")
                retriever = vectorstore.as_retriever()

    #-----------------------------------------------------------------------------------------------------------------------------------------------
                from langchain.prompts import ChatPromptTemplate
                from langchain_core.prompts import MessagesPlaceholder
                prompt_search_query = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                ("user","{input}"),
                ("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
                ])
                #retriever_chain = create_history_aware_retriever(llm_nvidia, retriever, prompt_search_query)
    #------------------------------------------------------------------------------------------------------------------------------------------------
                #prompt_get_answer = ChatPromptTemplate.from_messages([
                #("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
                #MessagesPlaceholder(variable_name="chat_history",)
                #("user","{input}"),
                #])

                from langchain.chains.combine_documents import create_stuff_documents_chain
                #document_chain=create_stuff_documents_chain(llm_nvidia,prompt_get_answer)
    #------------------------------------------------------------------------------------------------------------------------------------------------

                from langchain.chains import create_retrieval_chain
                #retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
                
    #------------------------------------------------------------------------------------------------------------------------------------------------
                from langchain.schema.output_parser import StrOutputParser
                from operator import itemgetter
                from langchain.schema.runnable import RunnablePassthrough
                import asyncio
                from langchain.memory import ConversationBufferMemory
                RAG_PROMPT = """\
                Use the following context and conversation history to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.
                
                Conversation History:
                {history}
                
                Question:
                {question}
                
                Context:
                {context}
                """
                
                # Initialize the memory
                memory = ConversationBufferMemory(memory_key="history", input_key="question", output_key="response")
                
                rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
                
                async def run_chain():
                    # Define the chain with memory
                    retrieval_augmented_generation_chain = (
                        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
                        | RunnablePassthrough.assign(context=itemgetter("context"))
                        | {"response": rag_prompt | llm_nvidia, "context": itemgetter("context")}
                    )
                
                    # Await the result of the async chain with memory
                    results = await retrieval_augmented_generation_chain.ainvoke(
                        {"question": user_input, "context": splits, "history": memory.chat_memory}
                    )
                
                    # Add the latest response to memory
                    memory.chat_memory.append({"question": user_input, "response": results})
                
                    return results
			
                with st.spinner("Retreving..."):
                    results = asyncio.run(run_chain())

    #------------------------------------------------------------------------------------------------------------------------------------------------
                system_prompt = (
                "Use the following context to answer the user's query."
                "Use three sentences maximum and answer in regress manner."
                "\n\n"
                "{context}"
                )

                from langchain.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "Use the following context to answer the user's query."
                        "Use three sentences maximum and answer in regress manner."
                        "\n\n"
                        "{context}"),
                        ("human", "{input}"),
                    ]
                )

                from langchain.chains.combine_documents import create_stuff_documents_chain
                question_answer_chain = create_stuff_documents_chain(llm_nvidia, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                #results = rag_chain.invoke({"input": user_input,  "context": splits}
                #st.write(results['answer'])

                #st.write(results)
                import time
                def stream_data():
                    for word in results["response"].content.split(" "):
                        yield word + " "
                        time.sleep(0.02)
                st.write_stream(stream_data)
                #st.write(results["response"].content)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
