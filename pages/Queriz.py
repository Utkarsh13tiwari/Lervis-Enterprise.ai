
import chromadb

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


ICON_BLUE = "https://github.com/Utkarsh13tiwari/Lervis-Enterprise.ai/blob/main/transperent_logo.png"

st.logo(ICON_BLUE, icon_image=ICON_BLUE)

# Set the page configuration
st.set_page_config(
    page_title="Queriz Enterprise",
    layout="wide",
    page_icon = ICON_BLUE
) 

st.markdown(
    """
    <style>
    .st-emotion-cache-qcpnpn{
        backgraound-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
groq = st.secrets.db_credentials.groq


class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.model.encode(d, batch_size=64).tolist() for d in documents]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()

embedding_model = CustomEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,  # Limiting the number of tokens per request
    timeout=None,
    max_retries=2,
)

search = TavilySearchAPIWrapper(tavily_api_key=os.environ["TAVILY_API_KEY"])
tavily_tool = TavilySearchResults(api_wrapper=search)
tools = [tavily_tool]

report_prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""You are a highly skilled Business Analyst, specializing in business governance, analysis, and development. Based on the provided documents, generate a detailed report covering the following:

    1. **Main Topic**: Identify the primary focus of the documents.
    2. **Key Components**: Summarize the important elements and aspects discussed in the document.
    3. **Business Analysis**: Analyze the information provided, highlighting any potential areas of risk, profit, or loss.
    4. **Performance Review**: Assess the current business performance based on the content.
    5. **Optional Suggestions**: If requested by the user, provide actionable recommendations for improving business performance.

    Use proper markdowns and points.

    Here are the documents: {input}

    Agent Scratchpad: {agent_scratchpad}

    Please provide the report below:
    """
)

agent = create_tool_calling_agent(llm, tools, report_prompt_template)
memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=1)
parser = PydanticOutputParser(pydantic_object=agent)
agent_exe=AgentExecutor(agent=agent, tools=tools, verbose=True ,stream_runnable=False, parser = parser)

    #-----------------------------------------------------------------------------------------------------------------

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'report_generated' not in st.session_state:
    st.session_state['report_generated'] = False
    st.session_state['report_data'] = None

if 'toggle' not in st.session_state:
    st.session_state.toggle = True

#-----------------------------------------------------------------------------------------------------------------
row1col1, row1col2 = st.columns(2)
row2col1, row2col2 = st.columns(2)

with row1col1:
    container1 = st.container(border=True)
    container1.header("QuerizRGPT")

    uploaded_file = container1.file_uploader("Upload a document")


    if uploaded_file is not None:

        #if st.session_state['report_generated']:
        #    container.write(st.session_state['report_data'])
        # Save the uploaded file temporarily
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getbuffer())

        
        loader = PyMuPDFLoader("temp_file")
        document = loader.load()

        # Step 3: Split the Document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(document)

        from langchain_chroma import Chroma
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection("langchain")

        vectorstore = Chroma(
            client=persistent_client,
            collection_name="langchain",
            embedding_function=embedding_model,
        )
        #vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="./chroma_langchain_db")
        retriever = vectorstore.as_retriever()

        from langchain_openai import ChatOpenAI
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from langchain_groq import ChatGroq

        def count_tokens(text):
            return len(text.split())  # Basic estimate by splitting on whitespace

        # Function to invoke the LLM in chunks and combine responses
        def generate_report_in_chunks(docs, token_limit=1000):
            report_responses = []
            current_tokens = 0
            current_batch = []

            for chunk in docs:
                chunk_token_count = count_tokens(chunk.page_content)
                # Check if adding this chunk exceeds the token limit
                if current_tokens + chunk_token_count > token_limit:
                    # If so, invoke the LLM for the current batch and reset
                    report = report_prompt_template.format(input=current_batch,agent_scratchpad="")
                    print("-----------------------------------------------------------------------")
                    print(f"Invoking LLM with {len(report)} tokens")
                    print("-----------------------------------------------------------------------")

                    try:
                        with st.spinner("Generating Report..."):
                            response = agent_exe.invoke({
                                "input": report,
                                "agent_scratchpad": ""  # This will be populated by the model's response.
                            })

                        output = response['output']
                        output_dict = json.loads(output)

                        # Check if the output indicates that Tavily is pending
                        if output.get("tool_call", {}).get("id") == "pending":
                            print("Tavily tool is pending, switching to LLM.")
                            raise TimeoutError("Tavily tool is in pending state.")

                        output = json.dumps(output_dict)
                        response = output

                    except (TimeoutError, KeyError, AttributeError) as e:
                        print(f"Tavily tool failed or returned an error: {e}. Switching to LLM directly.")
                        
                        try:
                            with st.spinner("Generating Report..."):
                                response = llm.invoke(report)
                            response = response.content
                        except Exception as e:
                            error_message = str(e)
                            if "rate limit" in error_message.lower():
                                wait_time_match = re.search(r'try again in (\d+)m(\d+.\d+)s|try again in (\d+)s', error_message)
                                if wait_time_match:
                                    if wait_time_match.group(1) and wait_time_match.group(2):
                                        # Case: Wait time is given in minutes and seconds
                                        wait_minutes = int(wait_time_match.group(1))
                                        wait_seconds = float(wait_time_match.group(2))
                                        total_wait_time = wait_minutes * 60 + wait_seconds
                                        st.write("Waiting.........................")
                                    elif wait_time_match.group(3):
                                        # Case: Wait time is given only in seconds
                                        st.write("Waiting.........................")
                                        total_wait_time = float(wait_time_match.group(3))
                                    
                                    print(f"Rate limit reached, waiting for {total_wait_time:.2f} seconds before retrying.")
                                    time.sleep(total_wait_time)  # Wait for the specified time
                                    with st.spinner("Generating Report..."):
                                        response = llm.invoke(report)
                                    response = response.content
                                else:
                                    print("Could not parse wait time, using default wait time of 60 seconds.")
                                    time.sleep(60)  # Fallback wait timee
                                    with st.spinner("Generating Report..."):
                                        response = llm.invoke(report)
                                    response = response.content
                            else:
                                print(f"An unexpected error occurred while invoking LLM: {e}")

                    except Exception as e:
                        print(f"An unexpected error occurred: {e}. Switching to LLM directly.")
                        # Fallback to LLM for any other unexpected exceptions
                        try:
                            with st.spinner("Generating Report..."):
                                response = llm.invoke(report)
                            response = response.content
                        except Exception as e:
                            error_message = str(e)
                            if "rate limit" in error_message.lower():
                                # Extract wait time from the error message
                                wait_time_match = re.search(r'try again in (\d+)m(\d+.\d+)s|try again in (\d+)s', error_message)
                                if wait_time_match:
                                    if wait_time_match.group(1) and wait_time_match.group(2):
                                        # Case: Wait time is given in minutes and seconds
                                        wait_minutes = int(wait_time_match.group(1))
                                        wait_seconds = float(wait_time_match.group(2))
                                        total_wait_time = wait_minutes * 60 + wait_seconds
                                        st.write("Waiting.........................")
                                    elif wait_time_match.group(3):
                                        # Case: Wait time is given only in seconds
                                        st.write("Waiting.........................")
                                        total_wait_time = float(wait_time_match.group(3))
                                    
                                    print(f"Rate limit reached, waiting for {total_wait_time:.2f} seconds before retrying.")
                                    time.sleep(total_wait_time)  # Wait for the specified time
                                    with st.spinner("Generating Report..."):
                                        response = llm.invoke(report)
                                    response = response.content
                                else:
                                    print("Could not parse wait time, using default wait time of 60 seconds.")
                                    time.sleep(60)  # Fallback wait time
                                    with st.spinner("Generating Report..."):
                                        response = llm.invoke(report)
                                    response = response.content
                            else:
                                print(f"An unexpected error occurred while invoking LLM: {e}")

                    def stream_data():
                        for word in response.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    container2.write_stream(stream_data)
                    report_responses.append(response)

                    current_batch = [chunk.page_content]
                    current_tokens = chunk_token_count
                else:
                    current_batch.append(chunk.page_content)
                    current_tokens += chunk_token_count

            if current_batch:
                report = report_prompt_template.format(input=current_batch,agent_scratchpad="")
                try:
                    with st.spinner("Generating Report..."):
                        response = agent_exe.invoke({
                            "input": report, 
                            "agent_scratchpad": "" 
                        })

                    output = response['output']
                    output_dict = json.loads(output)
                    if output.get("tool_call", {}).get("id") == "pending":
                        print("Tavily tool is pending, switching to LLM.")
                        raise TimeoutError("Tavily tool is in pending state.")
                    output = json.dumps(output_dict)
                    response = output

                except (TimeoutError, KeyError, AttributeError) as e:
                    print(f"Tavily tool failed or returned an error: {e}. Switching to LLM directly.")
                    with st.spinner("Generating Report..."):
                        response = llm.invoke(report)
                    response = response.content

                except Exception as e:
                    print(f"An unexpected error occurred: {e}. Switching to LLM directly.")
                    with st.spinner("Generating Report..."):
                        response = llm.invoke(report)
                    response = response.content

                def stream_data():
                    for word in response.split(" "):
                        yield word + " "
                        time.sleep(0.02)
                container2.write_stream(stream_data)
                report_responses.append(response)

            return " ".join(report_responses)

with row2col1:
    container2 = st.container(border=True)
    if uploaded_file is not None and container1.button("Generate Report"):
        with container1:
            report = generate_report_in_chunks(chunks)
        st.session_state['report_generated'] = True
        st.session_state['report_data'] = report
        st.download_button("Download Report", report)
    
    else:
        if st.session_state['report_generated']:
            def stream_data():
                for word in st.session_state['report_data'].split(" "):
                    yield word + " "
                    time.sleep(0.02)
            container1.write(st.session_state['report_data'])
            st.download_button("Download Report", st.session_state['report_data'])


with row1col2:
    container1A = st.container(border=True)
    container2A = st.container(border=True)
    container1A.header("QuerizGPT")

    use_custom_format = container1A.checkbox("Use custom response format")

    if use_custom_format:
        # Display text area to accept custom format from the user
        custom_format = container1A.text_area("Enter the custom response format:")

    with row1col2:
        user_query = container1A.chat_input("Ask your question about the report or follow ups:")

        with row1col2 and container1A:
            if user_query:
                if uploaded_file:

                    if st.session_state['report_generated']:
                        total_tokens = len(st.session_state['report_data'])
                        report = st.session_state['report_data']

                        if total_tokens > 3000:
                            st.write("Cannot  retrieve more than 3000 tokens. Please refine your query.")
                        
                        else:
                            container1A.write("You can now ask questions related to the generated report.")
                        
                    if st.session_state['report_generated'] and total_tokens < 3000:
                        if use_custom_format and custom_format:
                            query_prompt_template = PromptTemplate(
                                input_variables=["query", "report", "custom_format"],
                                template=f"""
                                As a Business Analyst, you should provide responses in the following format specified by the user:
                                
                                {custom_format}

                                Question: {{query}}

                                Report: {{report}}

                                Please respond in the format provided:
                                """
                            )
                        else:
                            query_prompt_template = PromptTemplate(
                                input_variables=["query", "report"],
                                template="""
                                As a Business Analyst, your role is to provide clear, concise, and insightful responses based on the following business report. 

                                Question: {query}

                                Report: {report}

                                Please provide a detailed answer below, including any relevant business analysis, potential impacts, and if applicable, suggestions to address the query:
                                """
                            )

                        query_chain = query_prompt_template | llm

                        query_response = query_chain.invoke({"query": user_query, "report": report})
                        agent_response = query_response.content

                        def stream_data():
                            for word in agent_response.split(" "):
                                yield word + " "
                                time.sleep(0.02)
                        st.write_stream(stream_data)
                
                else:
                    print("Executing normal queary 2")
                    if use_custom_format and custom_format:

                        simple_llm_prompt = PromptTemplate(
                            input_variables=["user_query", "custom_format"],
                            template=f"""
                            Respond to the user's query in the following format specified by the user:
                            
                            {custom_format}

                            Question: {{user_query}}

                            Provide a business-relevant response in the format provided:
                            """
                        )

                    else:
                        simple_llm_prompt = PromptTemplate(
                            input_variables=["user_query"],
                            template="""
                            Based on the provided business documents, please answer the following question: {user_query}.

                            If applicable, provide related YouTube links or websites for further learning or business insights.

                            If  you cannot find the answer, please let me know and I will assist you further.

                            You will only answer business queries and insights and nothing apart from this.

                            Note: Due to the size of the documents, this is a summarized response, focusing on the most relevant information.
                            """
                        )

                    chain = simple_llm_prompt | llm 
                    llm_response = chain.invoke({"user_query": user_query})
                    agent_response = llm_response.content
                    def stream_data():
                        for word in llm_response.content.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    st.write_stream(stream_data)

                message = {'user': user_query, 'AI': agent_response}
                st.session_state.chat_history.append(message)
        if os.path.exists("temp_file.pdf"):
            os.remove("temp_file.pdf")
