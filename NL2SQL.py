import os
import time
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
import streamlit as st
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import pandas as pd
import json


#--------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ICON_BLUE = "transperent_logo.png"

st.logo(ICON_BLUE, icon_image=ICON_BLUE)

# Set the page configuration
st.set_page_config(
    page_title="Lervis Enterprice",
    layout="wide",
    page_icon = ICON_BLUE
)  

st.markdown(
    """
    <style>
    .message-container {
        margin-bottom: 20px;
        width: 100%;
    }
    .message-container .user-message {
        text-align: right;
        padding: 10px;
        border-radius: 5px;
        background-color: #0c0912;
        margin-bottom: 20px;
    }
    .message-container .assistant-message {
        text-align: left;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .st-emotion-cache-qdbtli {
        width: 70%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#-------------------------------------------  API Key's ---------------------------------------------------------------
openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
groq = st.secrets.db_credentials.groq
nvidia = st.secrets.db_credentials.nvidia

#----------------------------------------------------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 4, 1])
#------------------------------------------------- llm's -----------------------------------------------------------------------------
llm_gpt = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key = openai)

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,  # Limiting the number of tokens per request
    timeout=None,
    max_retries=2,
    api_key = groq
)
#-------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------- db --------------------------------------------------------------------------------------------------------------------------

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
col1, col2, col3 = st.columns([1,4,1])
with col1:
        st.image('transperent_logo.png', width=200)

with col3:
    with st.popover("Usage"):
        st.markdown(
            """
	        <h1>SQL Query Bot</h1>
            <p>This bot is designed to assist users in answering SQL-related questions by executing queries on a database and then interpreting the results. Here's how it works:</p>
            <h2>Core Functionality</h2>
            <ol>
                <li><strong>SQL Query Execution</strong>: The bot takes the user's question, generates or retrieves a corresponding SQL query, and executes it on a connected database.</li>
                <li><strong>Result Interpretation</strong>: After executing the query, it interprets the SQL results and provides a response in clear, human-readable language.</li>
                <li><strong>Customizable Output Format</strong>:
                    <ul>
                        <li><strong>SQL Query</strong>: It always shows the SQL query used for transparency, wrapped in a <code>sql</code> block so that the user can review it.</li>
                        <li><strong>Answer</strong>: It provides the query results as strings to answer the user's question directly.</li>
                    </ul>
                </li>
            </ol>
            <h2>Data Visualizations</h2>
            <p>If the user asks for a table or chart, the bot can convert the SQL results into a structured JSON format to display a table, bar chart, or line chart.</p>
            <p><strong>Data formats include:</strong></p>
            <ul>
                <li><strong>Tables</strong>: Display SQL results in a tabular format with columns and data.</li>
                <li><strong>Bar Charts</strong>: Present the data in a bar chart format for better visual comparison.</li>
                <li><strong>Line Charts</strong>: Display trends or patterns in data over time or another axis using a line chart.</li>
            </ul>
            <h2>Example Flow</h2>
            <ol>
                <li><strong>User Question</strong>: "What are the top 5 best-selling products?"</li>
                <li><strong>Bot Response</strong>:
                    <ul>
                        <li><strong>SQL query used</strong>: Displays the exact SQL query executed.</li>
                        <li><strong>Answer</strong>: Provides the textual answer, like the names and sales numbers of the top 5 products.</li>
                        <li>If requested:
                            <ul>
                                <li><strong>Data format (table)</strong>: Shows a table with the product names and sales numbers.</li>
                                <li><strong>Data format (bar chart)</strong>: Provides a bar chart visualization with the products and sales figures.</li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ol>
            <h2>Key Benefits</h2>
            <ul>
                <li>Automates SQL query execution and result interpretation.</li>
                <li>Allows users to ask complex SQL-based questions without needing to know SQL syntax.</li>
                <li>Provides clear, visual representations of data when requested, making it easier to understand the results.</li>
            </ul>
            <p>In short, this bot acts as an intelligent assistant that not only executes SQL queries but also helps visualize and interpret the data, making it a powerful tool for querying databases.</p>
            """, 
            unsafe_allow_html=True)


#----------------------------------------------- O&A -----------------------------------------------------------
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm_gpt, db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    
    Response Format:
    **SQL query used**: Display the exact SQL query that you executed with in ```sql ```.
    **Answer**: Provide the results (in strings) based on the query execution.

    If only only if asked for ploting or table then also provide Data format as:
    **Data format**: If the user requires or asked to drawing/show a table, reply as follows:
                     {{"table": {{"columns": ["column1", "column2", ...], "data": [["value1", "value2", ...], ["value1", "value2", ...], ...]}}}}

                     If the query requires or asked to create/draw/show a bar chart/plot/diagram, reply as follows:
                     {{"bar": {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}}}

                     If the query requires or asked to create/draw/show a line chart/plot/diagram, reply as follows:
                     {{"line": {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}}}

                     with in ```json ```.

    There can only be two types of chart, "bar" and "line".
    
    Return all output as a string.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm_gpt
    | StrOutputParser()
)
#----------------------------------- System prompt ----------------------------------------------------------

toolkit = SQLDatabaseToolkit(db=db, llm=llm_gpt)

tools = toolkit.get_tools()


SQL_PREFIX = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

To start, you should ALWAYS look at the tables in the database to see what you can query. 
Do NOT skip this step. Then you should query the schema of the most relevant tables.

Response Format:
1. **SQL query used**: Display the exact SQL query that you executed.
2. **Answer**: Provide the results based on the query execution.
"""


system_message = SystemMessage(content=SQL_PREFIX)
#agent_executor = create_react_agent(llm_gpt, tools, messages_modifier=system_message)

#--------------------------------------------- Agent --------------------------------------------------------

#agent_executor = create_sql_agent(llm_gpt, db=db, agent_type="openai-tools", verbose=True, message = system_message)


#--------------------------------------------------------------

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app and visualize data as needed.
    
    Args:
        response_dict: The response from the agent in JSON format.
    
    Returns:
        None
    """
    
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])
    
    # Check if the response contains a bar chart.
    if "bar" in response_dict:
        bar_data = response_dict["bar"]
        columns = bar_data["columns"]
        data = bar_data["data"]

        # Convert the columns and data into a Pandas DataFrame.
        try:
            df = pd.DataFrame(data=[data], columns=columns)
            st.write("Bar Chart DataFrame created:")
            st.write(df)  # Display the DataFrame
            
            # Plot the bar chart using Streamlit.
            st.bar_chart(df.T)  # Transpose for proper column-bar mapping

        except ValueError as e:
            st.error(f"Error creating DataFrame for bar chart: {e}")
    
    # Check if the response contains a line chart.
    if "line" in response_dict:
        line_data = response_dict["line"]
        columns = line_data["columns"]
        data = line_data["data"]
        
        # Convert to DataFrame.
        try:
            df = pd.DataFrame(data=[data], columns=columns)
            st.write("Line Chart DataFrame created:")
            st.write(df)  # Display the DataFrame
            
            # Plot the line chart using Streamlit.
            st.line_chart(df.T)  # Transpose for proper x-axis mapping

        except ValueError as e:
            st.error(f"Error creating DataFrame for line chart: {e}")
    
    # Check if the response contains a table.
    if "table" in response_dict:
        table_data = response_dict["table"]
        columns = table_data["columns"]
        data = table_data["data"]
        
        # Convert the columns and data into a Pandas DataFrame.
        try:
            df = pd.DataFrame(data, columns=columns)
            st.write("Table DataFrame created:")
            st.table(df)  # Display the DataFrame as a table

        except ValueError as e:
            st.error(f"Error creating DataFrame for table: {e}")


def decode_response(response):
    # Check if the response is empty or None
    if not response:
        print("Response is empty or None.")
        return None

    # Check if response is a valid JSON string
    try:
        import json
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON. Error: {e}")
        print("Response content:", response)  # Print the response for debugging
        return None

# Frontend code. Initialising states ---------------------------------------------------------------------------------------------------------------------------------------
if 'Session' not in st.session_state:
    st.session_state['Session'] = {'Session 1': []}

if 'run' not in st.session_state:
    st.session_state.run = False

for conv_id in st.session_state['Session']:
    if st.sidebar.button(conv_id):
        st.session_state['current_session'] = conv_id
                         
if st.sidebar.button("Start New Session"):
    new_conv_id = f"Session {len(st.session_state['Sessions']) + 1}"
    st.session_state['Session'][new_conv_id] = []
    st.session_state['current_Session'] = new_conv_id  # Set the new Session as the current one 

# Set the default current conversation
if 'current_Session' not in st.session_state:
    st.session_state['current_Session'] = 'Session 1'

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


#-------------------------------------------------------------------------------------------------------------------------------------
# Use columns to center the conversation container and set width

with col2:
    into_text = "### Chat with your SQL database"
    with col2:
        def stream_data():
            for word in into_text.split(" "):
                yield word + " "
                time.sleep(0.08)
        st.write_stream(stream_data)
    st.write(f"### {st.session_state['current_Session']} :")
    parser = StrOutputParser()
    # Display the conversation history
    for message in st.session_state['Session'][st.session_state['current_Session']]:
        #message_class = "user-message" if message["isUser"] else "assistant-message"

        if message["isUser"]:
            message_class = "user-message"
            st.markdown(f'<div class="message-container"><p class="{message_class}">{message["text"]}</p></div>', unsafe_allow_html=True)

        elif not message["isUser"]: 
            message_class = "assistant-message"
            st.write("### Agent Response:")
            st.write(parser.parse(message["text"]))
#--------------------------------------------------------------------------------------------------------
# Input field for user query
user_input = st.chat_input("Query your database:",args=(True,))

#----------------------------------------------------------------------------------------------------------
if user_input:

    with col2:
        st.write(f'<div class="message-container"><p class="user-message">{user_input}</p></div>', unsafe_allow_html=True)

    with col2:
        with st.spinner("Agent"):
            #agent_response = agent_executor.invoke(user_input)
            #agent_response = agent_executor.invoke({"messages": [HumanMessage(content="Which country's customers spent the most?")]})
            agent_response = chain.invoke({"question": user_input})
            print("**********************************************************************************")
            print(agent_response)
            print("**********************************************************************************")

        response = agent_response
        #if agent_response:
        #    decoded_response = decode_response(agent_response)


        #agent_response = write_response(agent_response)
        #agent_response = agent_response["output"]
        #st.write(agent_response)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #print(agent_response)
        print("**********************************************************************************")

        st.write("### Agent Response:")
        def stream_data():
            for word in agent_response.split(" "):
                yield word + " "
                time.sleep(0.02)
        st.write_stream(stream_data)

#------------------------- plotting --------------------------------------------------------------------------------
        try:
            if '"bar": {' in response:

                start_index = response.index('{"bar":')
                json_str = response[start_index:]

                # Find the closing brace for the "bar" object
                end_index = json_str.rindex('}') + 1  # Find the last closing brace
                json_str = json_str[:end_index]

                # Print the extracted JSON string for debugging
                print("Extracted JSON string:", json_str)

                # Convert the JSON string to a dictionary
                try:
                    json_data = json.loads(json_str)
                    
                    # Create a Pandas DataFrame from the extracted data
                    df = pd.DataFrame(json_data['bar']['data'], columns=json_data['bar']['columns'])

                    print("Data format found. DataFrame created:")
                    print(df)
                    st.bar_chart(df.set_index(df.columns[0]),use_container_width=True)
                except json.JSONDecodeError as e:
                    print("Failed to decode JSON:", e)

            if '"line": {' in response:

                start_index = response.index('{"line":')
                json_str = response[start_index:]

                # Find the closing brace for the "bar" object
                end_index = json_str.rindex('}') + 1  # Find the last closing brace
                json_str = json_str[:end_index]

                # Print the extracted JSON string for debugging
                print("Extracted JSON string:", json_str)

                # Convert the JSON string to a dictionary
                try:
                    json_data = json.loads(json_str)
                    
                    # Create a Pandas DataFrame from the extracted data
                    df = pd.DataFrame(json_data['line']['data'], columns=json_data['line']['columns'])

                    print("Data format found. DataFrame created:")
                    print(df)
                    st.line_chart(df)
                except json.JSONDecodeError as e:
                    print("Failed to decode JSON:", e)

            if '"table": {' in response:

                start_index = response.index('{"table":')
                json_str = response[start_index:]

                # Find the closing brace for the "bar" object
                end_index = json_str.rindex('}') + 1  # Find the last closing brace
                json_str = json_str[:end_index]

                # Print the extracted JSON string for debugging
                print("Extracted JSON string:", json_str)

                # Convert the JSON string to a dictionary
                try:
                    json_data = json.loads(json_str)
                    
                    # Create a Pandas DataFrame from the extracted data
                    df = pd.DataFrame(json_data['table']['data'], columns=json_data['table']['columns'])

                    print("Data format found. DataFrame created:")
                    print(df)
                    st.table(df)
                except json.JSONDecodeError as e:
                    print("Failed to decode JSON:", e)

        except Exception as e:
            st.error(f"Unable to plot due to  error: {e}")

#------------------------- plotting end--------------------------------------------------------------------------------

    message = {'user': user_input, 'AI': agent_response}
    st.session_state.chat_history.append(message)

    with col2:
        st.session_state['Session'][st.session_state['current_Session']].append({"isUser": True, "text": user_input})
        st.session_state['Session'][st.session_state['current_Session']].append({"isUser": False, "text": agent_response})
