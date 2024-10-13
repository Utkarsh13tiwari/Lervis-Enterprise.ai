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


def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()

def create_agent(filename: str, filetype: str):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """

    # Create an OpenAI object.
    llm_gpt = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5,api_key = openai)

    # Read the CSV file into a Pandas DataFrame.
    #df = pd.read_csv(filename, encoding='latin1', sep='delimiter', header=None, engine='python')
    if filetype == 'csv':
        df = pd.read_csv(filename, dtype=str)
        st.sidebar.write(df.head(5))
    elif  filetype == 'xls':
        df = pd.read_excel(filename)
        st.sidebar.write(df.head(5))
    print("**************************************************************************************")
    print(df.shape)
    print("**************************************************************************************")

    CSV_agent = create_pandas_dataframe_agent(llm_gpt, df, verbose=False, handle_parsing_errors=True ,allow_dangerous_code=True)
    return CSV_agent

def decode_response(response):
    # Check if the response is empty or None
    if not response:
        print("Response is empty or None.")
        return None

    # Check if response is a valid JSON string
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON. Error: {e}")
        print("Response content:", response)  # Print the response for debugging
        return None

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        def stream_data():
                for word in response_dict["answer"].split(" "):
                    yield word + " "
                    time.sleep(0.02)
        st.write_stream(stream_data)
        #st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        # Extract columns and data from the response
        bar_data = response_dict["bar"]
        columns = bar_data["columns"]
        data = bar_data["data"]

        # Convert to DataFrame
        try:
            # Create DataFrame from the data and columns
            df = pd.DataFrame(data, columns=columns)
            st.write("DataFrame created:")
            st.write(df)  # Debugging step: display the DataFrame
            
            # Check if "Index" column exists before setting it as the index
            if "Index" in df.columns:
                df.set_index("Index", inplace=True)
                st.bar_chart(df)
            
            else:
                st.bar_chart(df)

        except ValueError as e:
            st.error(f"Error creating DataFrame: {e}")

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        columns = data["columns"]
        data = data["data"]
        try:
            # Create DataFrame from the data and columns
            df = pd.DataFrame(data, columns=columns)
            st.write("DataFrame created:")
            st.write(df)  # Debugging step: display the DataFrame
            
            # Check if "Index" column exists before setting it as the index
            if "Index" in df.columns:
                df.set_index("Index", inplace=True)
                st.line_chart(df)
            else:
                st.line_chart(df)

        except ValueError as e:
            st.error(f"Error creating DataFrame: {e}")
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

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
            #if data.type == 'application/vnd.ms-excel' or data.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                #st.write("Excel file uploaded.")
                #agent = create_agent(data, 'xls')

                # Query the agent.
                #with st.spinner("Retreving..."):
                    #response = query_agent(agent=agent, query=user_input)

                # Debug output
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                #print(response)
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                # Decode the response.
                #if response:
                #    decoded_response = decode_response(response)

                # Write the response to the Streamlit app.
                #if decode_response:
                #    write_response(decoded_response)
                
                #else:
                #    st.write(response)

            #if data.type == 'text/csv':
                #agent = create_agent(data, 'csv')

                # Query the agent.
                #response = query_agent(agent=agent, query=user_input)
                #st.write(response.answer)

                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                #print(response)
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                # Decode the response.
                #if response:
                #    decoded_response = decode_response(response)

                # Write the response to the Streamlit app.
                #if decode_response:
                #    write_response(decoded_response)
                
                #else:
                #    st.write(response)

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
                from operator import itemgetter
                from langchain.schema.output_parser import StrOutputParser
                from langchain.schema.runnable import RunnablePassthrough
                import asyncio

                RAG_PROMPT = """\
                Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

                Question:
                {question}

                Context:
                {context}
                """

                rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

                async def run_chain():
                    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

                    retrieval_augmented_generation_chain = (
                        {"context": itemgetter("question") 
                        | retriever, "question": itemgetter("question")}
                        | RunnablePassthrough.assign(context=itemgetter("context"))
                        | {"response": rag_prompt | llm_nvidia, "context": itemgetter("context")}
                    )

                    # Await the result of the async chain
                    results = await retrieval_augmented_generation_chain.ainvoke({"question" : user_input, "context": splits})
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
