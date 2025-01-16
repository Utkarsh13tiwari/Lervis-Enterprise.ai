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
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types  import AgentType

import pandas as pd
import json
from streamlit_pills import pills
import pandas as pd
from sqlalchemy import create_engine


#--------------------------------------------------------------------------------------------------------------------
openai = st.secrets.db_credentials.openai 
tavily = st.secrets.db_credentials.tavily
groq = st.secrets.db_credentials.groq
nvidia = st.secrets.db_credentials.nvidia
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ICON_BLUE = r"transperent_logo.png"

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

#----------------------------------------------------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 4, 1])
#------------------------------------------------- llm's -----------------------------------------------------------------------------
#llm_gpt = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai)

llm_groq = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,  # Limiting the number of tokens per request
    timeout=None,
    api_key=groq,
    max_retries=2,
)
#-------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------- db --------------------------------------------------------------------------------------------------------------------------

engine = create_engine(r"sqlite:///Chinook.db")

db = SQLDatabase.from_uri(r"sqlite:///Chinook.db")
col1, col2, col3 = st.columns([1,4,1])
linkedin_url = "https://www.linkedin.com/company/lervis-ai/?viewAsMember=true" 
with col1:
        st.image('transperent_logo.png', width=200)
        with col1:
            st.link_button("Connect", linkedin_url)

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
template = """
You are a plotting assistant. Generate a plot in Plotly format based on the LLM response. You must identfy the SQL code in the query and generate a plot of the response:
{query}
{df}

The plot should be a Plotly figure object like this:
fig = px.scatter(df, x="sepal_width", y="sepal_length")

You can also add other parameters to make it look visually appealing like this:

fig = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
)

Make sure when you join two columns in df, you just keep only 1 of the column name, example:

fig = px.bar(df, x="FirstName LastName", y="TotalSpent", title="Customer who spent most on Iron Maiden")
The above is join of two coloumns  "FirstName" and "LatsName" in df.
correct code is:
fig = px.bar(df, x="FirstName", y="TotalSpent", title="Customer who spent most on Iron Maiden")


Please Return only the plot generation code and do not return anything else.

return example:
fig = px.bar(df, x="FirstName", y="TotalSpent", title="Customer who spent most on Iron Maiden")
"""
import plotly.express as px
from langchain.chains import LLMChain
prompt = PromptTemplate(input_variables=["query"], template=template)
chain2 = LLMChain(llm=llm_groq, prompt=prompt)
#----------------------------------------------- O&A -----------------------------------------------------------
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm_groq, db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    If you dont know the answer just say you dont know.
    
    Response Format:
    **SQL query used**: Display the exact SQL query that you executed with in ```sql ```.
    **Answer**: Provide the results in markdown and points based on the query execution.
    
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

toolkit = SQLDatabaseToolkit(db=db, llm=llm_groq)

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
agent_executor = create_react_agent(llm_groq, tools, messages_modifier=system_message)
#agent_executor = AgentExecutor(agent=agent_executor, tools=tools)
#--------------------------------------------- Agent --------------------------------------------------------

prompt = PromptTemplate.from_template(SQL_PREFIX)
#agent_executor = create_sql_agent(llm_gpt, db=db, agent_type="openai-tools", verbose=True, message = system_message)
agent_executor = create_sql_agent(llm_groq, db=db, verbose=True, top_k=1000, prefix=prompt, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, return_intermediate_steps=True)

#--------------------------------------------------------------

# Frontend code. Initialising states ---------------------------------------------------------------------------------------------------------------------------------------

if 'Session' not in st.session_state:
    st.session_state['Session'] = {'Session 1': []}

if 'current_Session' not in st.session_state:
    st.session_state['current_Session'] = 'Session 1'

if 'run' not in st.session_state:
    st.session_state.run = False

# Sidebar for sessions
for conv_id in st.session_state['Session']:
    if st.sidebar.button(conv_id):
        st.session_state['current_Session'] = conv_id  # Switch to the selected session

if st.sidebar.button("Start New Session"):
    new_conv_id = f"Session {len(st.session_state['Session']) + 1}"
    st.session_state['Session'][new_conv_id] = []  # Initialize the new session
    st.session_state['current_Session'] = new_conv_id  # Set the new session as the current one

with col2:

    # Display the current session
    st.write(f"### {st.session_state['current_Session']} :")

    # Chat history for the current session
    if st.session_state['current_Session'] not in st.session_state['Session']:
        st.session_state['Session'][st.session_state['current_Session']] = []

    # Display the conversation history for the current session
    for message in st.session_state['Session'][st.session_state['current_Session']]:
        if message["isUser"]:
            st.markdown(f'<div class="message-container"><p class="user-message">{message["text"]}</p></div>', unsafe_allow_html=True)
        else:
            st.write("### Agent Response:")
            st.write(message["text"])
            # Check if there's an associated plot
            if "plot" in message:
                st.plotly_chart(message["plot"], use_container_width=True)

#--------------------------------------------------------------------------------------------------------
# Input field for user query
user_input = st.chat_input("Query your database:", args=(True,))

with col2:
    if len(st.session_state['Session'][st.session_state['current_Session']]) == 0:
        st.divider()
        st.markdown(
            """
            <div style="text-align: center;">
                <h3>Query your database in Natural Language</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        row1 = st.columns(2)
        row2 = st.columns(2)

        row1[0].container(height=130).markdown(
            """
            <div style="text-align: center; height: 100px; display: flex; align-items: center; justify-content: center;">
                Who is writing the rock music?
            </div>
            """, 
            unsafe_allow_html=True
        )

        row1[1].container(height=130).markdown(
            """
            <div style="text-align: center; height: 100px; display: flex; align-items: center; justify-content: center;">
                Which artist has earned the most according to the Invoice Lines? Use this artist to find which customer spent the most on this artist.
            </div>
            """, 
            unsafe_allow_html=True
        )

        row2[0].container(height=130).markdown(
            """
            <div style="text-align: center; height: 100px; display: flex; align-items: center; justify-content: center;">
                Who are our top Customers according to Invoices?
            </div>
            """, 
            unsafe_allow_html=True
        )

        row2[1].container(height=130).markdown(
            """
            <div style="text-align: center; height: 100px; display: flex; align-items: center; justify-content: center;">
                Which Employee has the Highest Total Number of Customers?
            </div>
            """, 
            unsafe_allow_html=True
        )

#------------------------------------------------------------------------------------------------------------------------------
with col2:
    if user_input:
        # Display the user input
        st.markdown(f'<div class="message-container"><p class="user-message">{user_input}</p></div>', unsafe_allow_html=True)

        with st.spinner("Agent is responding..."):
            # Simulate agent response
            agent_response = chain.invoke({"question": user_input})  # Invoke the agent (simulated)

            # Display agent response
            st.write("### Agent Response:")
            st.write(agent_response)

            # Append the user input and agent response to the current session's chat history
            st.session_state['Session'][st.session_state['current_Session']].append({"isUser": True, "text": user_input})
            st.session_state['Session'][st.session_state['current_Session']].append({"isUser": False, "text": agent_response})

            # Extract SQL from the agent response and execute it
            start_marker = "```sql" 
            end_marker = "```"
            start = agent_response.find(start_marker) + len(start_marker)
            end = agent_response.find(end_marker, start)
            query_text = agent_response[start:end].strip()

            from sqlalchemy import text
            with engine.begin() as conn:
                query = text(query_text)
                df = pd.read_sql(query, conn)

            # Generate and display plot
            try:
                plot_code = chain2.invoke({"query": user_input, "df": df})
                st.code(plot_code["text"], language="python")
                exec(plot_code["text"])  # Execute the code to generate a plot
                # Check if 'fig' was created
                if 'fig' in locals():
                    st.session_state['Session'][st.session_state['current_Session']].append({"isUser": False, "text": "Generated Plot:", "plot": fig})
                    st.plotly_chart(fig, key="user_generated_plot")
            except Exception as e:
                st.error(f"Could not plot  the data because: {e}")

            if len(st.session_state['Session'][st.session_state['current_Session']]) == 3:
                print(len(st.session_state['Session'][st.session_state['current_Session']]))
                st.rerun()



    # After each input, the messages are appended to the respective session’s chat history.
