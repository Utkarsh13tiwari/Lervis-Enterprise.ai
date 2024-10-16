import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
import plotly.express as px
import plotly.graph_objects as go
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from streamlit_pills import pills  
from langchain.chains import LLMChain


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


groq_llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None, 
    timeout=None,
    max_retries=2,
    api_key = groq
)


# # Initialize the OpenAI language model
openai_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai)

col1, col2, col3 = st.columns([1,4,1])
with col1:
        st.image('transperent_logo.png', width=200)

default_examples = [
    {"query": "What are the sales trends?", "response": "Sales increased by 10% in Q1.", "insight": "The company should focus on sustaining the growth by investing in marketing."},
    {"query": "What is the employee satisfaction?", "response": "Employee satisfaction is 85%.", "insight": "High employee satisfaction is a positive sign, but further surveys can help understand areas of improvement."}
]

def analyze_responses(examples, user_query, user_response):
    example_prompt = PromptTemplate.from_template(
        "Query: {query}\nResponse: {response}\nInsight: {insight}"
    )
    
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=(  
            '''You are an AI assistant specialized in analyzing natural language business queries and responses. 
            Your task is to thoroughly understand the given query and response, then generate 
            insightful, actionable, and strategic insights. Focus on:
            1. Identifying key trends or patterns
            2. Highlighting critical issues or opportunities
            3. Suggesting data-driven recommendations
            4. Providing context for decision-making
            5. Proposing next steps or areas for further investigation'''
        ),
        suffix="Query: {query}\nResponse: {response}\nInsight: ",
        input_variables=["query", "response"],
    )

    final_prompt = prompt.format(query=user_query, response=user_response)
    result = groq_llm.invoke(final_prompt)
    return result.content.strip()

# Function to query CSV data
def query_csv(df, query):
    info = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'summary': df.describe().to_dict()
    }
    
    prompt = f"""
    You are an AI assistant specialized in analyzing CSV data. Given the following data summary and natural language query, provide a comprehensive and accurate response.

    CSV Data Summary:
    Columns: {info['columns']}
    Shape: {info['shape']}
    Data Types: {info['dtypes']}
    Summary Statistics: {info['summary']}
    
    Here's a sample of the data (first 5 rows):
    {df.head().to_string()}

    Natural Language Query: {query}
    Analyze the provided information and respond to the natural language query. Use specific numbers and trends from the available data.
    """
    
    result = groq_llm.invoke(prompt)
    return result.content.strip()

# Define the prompt template for visualization
template = """
You are a plotting assistant. Generate a plot in Plotly format based on the following query:
{query}

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
Return only the plot generation code.
"""

prompt = PromptTemplate(input_variables=["query"], template=template)
chain = LLMChain(llm=openai_llm, prompt=prompt)


with col2:
    st.title("👨‍💻 AI-Powered CSV Query and Insight Generator")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("CSV file uploaded successfully!")
    st.write(df.head())

    selected = pills("Select an option:", ["Generate Insights", "Plot Graph"])

    

    if selected == "Generate Insights":
        # Show the checkbox and example input only when "Generate Insights" is selected
        use_custom_format = st.checkbox("Use custom few-shot examples")
        examples = default_examples

        if use_custom_format:
            st.write("Provide your own examples in the format: 'query => response => insight'")
            example_input = st.text_area(
                "Enter few-shot examples (one per line, format: 'query => response => insight')"
            )
            if example_input:
                examples = [
                    {"query": q, "response": r, "insight": i}
                    for q, r, i in [line.split(" => ") for line in example_input.splitlines() if " => " in line]
                ]
        user_query = st.text_area("Enter your prompt:")

        if st.button("Generate Insights"):
            if user_query:
                with st.spinner("Analyzing CSV data and generating insights..."):
                    csv_response = query_csv(df, user_query)
                    st.subheader("Query Response:")
                    st.write(csv_response)

                    insight = analyze_responses(examples, user_query, csv_response)
                    st.subheader("Generated Insight:")
                    st.write(insight)

    elif selected == "Plot Graph":
        user_query = st.text_area("Enter your prompt:")
        if st.button("Generate Plot"):
            if user_query:
                plot_code = chain.run(user_query)
                
                editable_plot_code = st.text_area("Edit your plot code:", value=plot_code, height=200)                
                try:
                    local_scope = {'df': df, 'px': px}
                    exec(editable_plot_code, local_scope)  # Execute the code
                    if 'fig' in local_scope:
                        st.plotly_chart(local_scope['fig'], key="user_generated_plot", use_container_width=True)
                    else:
                        st.error("No figure was generated. Please check your code.")
                except Exception as e:
                    st.error(f"An error occurred while generating the plot: {e}")



    
