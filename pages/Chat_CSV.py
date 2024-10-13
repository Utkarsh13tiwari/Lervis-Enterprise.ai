import os
import streamlit as st
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import sqldf

col1, col2, col3 = st.columns([1,4,1])
with col1:
        st.image(r'C:\Users\utkar\OneDrive\Desktop\Lervis Enterprise\transperent_logo.png', width=200)
# AI Decision Tracker Class
class AIDecisionTracker:
    def __init__(self):
        self.tracking_df = pd.DataFrame(columns=[
            'timestamp',
            'query',
            'response',
            'insight',
            'decision_made',
            'predicted_impact',
            'actual_impact',
            'metric_name',
            'baseline_value',
            'current_value',
            'percentage_change'
            
        ])
    def create_timeline_chart(self):
        """Creates a timeline chart of decisions and their impacts"""
        if len(self.tracking_df) == 0:
            return None
            
        fig = px.scatter(
            self.tracking_df,
            x='timestamp',
            y='percentage_change',
            color='actual_impact',
            size='current_value',
            hover_data=['metric_name', 'decision_made'],
            title='Decision Impact Timeline',
            labels={
                'timestamp': 'Time',
                'percentage_change': 'Impact (%)',
                'actual_impact': 'Impact Type'
            }
        )
        return fig
    
    def create_metric_performance_chart(self):
        """Creates a bar chart showing performance by metric"""
        if len(self.tracking_df) == 0:
            return None
            
        metric_summary = self.tracking_df.groupby('metric_name').agg({
            'percentage_change': 'mean',
            'actual_impact': lambda x: (x == 'Positive').mean() * 100
        }).reset_index()
        
        fig = go.Figure(data=[
            go.Bar(
                name='Average Impact (%)',
                x=metric_summary['metric_name'],
                y=metric_summary['percentage_change'],
                marker_color='lightblue'
            ),
            go.Bar(
                name='Success Rate (%)',
                x=metric_summary['metric_name'],
                y=metric_summary['actual_impact'],
                marker_color='lightgreen'
            )
        ])
        
        fig.update_layout(
            title='Performance by Metric',
            barmode='group',
            xaxis_title='Metric',
            yaxis_title='Percentage'
        )
        return fig
    
    def create_impact_distribution_pie(self):
        """Creates a pie chart showing the distribution of impact types"""
        if len(self.tracking_df) == 0:
            return None
            
        impact_counts = self.tracking_df['actual_impact'].value_counts()
        fig = px.pie(
            values=impact_counts.values,
            names=impact_counts.index,
            title='Distribution of Decision Impacts',
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'None': 'grey'}
        )
        return fig
    
    def create_decision_heatmap(self):
        """Creates a heatmap showing decision activity over time"""
        if len(self.tracking_df) == 0:
            return None
            
        self.tracking_df['date'] = pd.to_datetime(self.tracking_df['timestamp']).dt.date
        daily_counts = self.tracking_df.groupby('date').size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        fig = px.density_heatmap(
            daily_counts,
            x='date',
            y=None,
            z='count',
            title='Decision Activity Heatmap',
            labels={'date': 'Date', 'count': 'Number of Decisions'}
        )
        return fig
        
    def add_decision(self, query, response, insight, decision_made, predicted_impact, metric_name, baseline_value):
        new_record = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'insight': insight,
            'decision_made': decision_made,
            'predicted_impact': predicted_impact,
            'actual_impact': None,
            'metric_name': metric_name,
            'baseline_value': baseline_value,
            'current_value': baseline_value,
            'percentage_change': 0.0
        }
        self.tracking_df = pd.concat([self.tracking_df, pd.DataFrame([new_record])], ignore_index=True)
        
    def update_impact(self, decision_id, current_value):
        if decision_id in self.tracking_df.index:
            baseline = self.tracking_df.loc[decision_id, 'baseline_value']
            self.tracking_df.loc[decision_id, 'current_value'] = current_value
            self.tracking_df.loc[decision_id, 'percentage_change'] = ((current_value - baseline) / baseline) * 100
            self.tracking_df.loc[decision_id, 'actual_impact'] = 'Positive' if current_value > baseline else 'Negative'
            
    def get_impact_summary(self):
        return {
            'total_decisions': len(self.tracking_df),
            'positive_impacts': len(self.tracking_df[self.tracking_df['actual_impact'] == 'Positive']),
            'average_improvement': self.tracking_df['percentage_change'].mean() if len(self.tracking_df) > 0 else 0.0,
            'metrics_tracked': self.tracking_df['metric_name'].unique().tolist()
        }

# Set the API key for NVIDIA language model
nvidia = st.secrets.db_credentials.nvidia

# Initialize the language model
@st.cache_resource
def load_llm():
    return ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", api_key = nvidia)

llm = load_llm()

# [Keep all the existing functions from the first code: initial_examples, is_valid_json, parse_custom_examples, analyze_responses, query_csv]

# Define initial few-shot examples
initial_examples = [
    {
        "query": "What were our top-selling products last quarter?",
        "response": "Our top-selling products were smartphones (35% of sales), laptops (25%), and smart home devices (15%).",
        "insight": "Electronics dominate sales. Consider expanding the range of smart home devices to capitalize on growing market interest."
    },
    {
        "query": "How has customer satisfaction changed since implementing the new support system?",
        "response": "Customer satisfaction scores increased from 7.5 to 8.2 out of 10 in the three months since implementation.",
        "insight": "The new support system has positively impacted customer satisfaction. Analyze specific features driving this improvement for potential expansion to other areas."
    },
    {
        "query": "What's the current employee turnover rate compared to last year?",
        "response": "The current employee turnover rate is 15%, down from 18% last year.",
        "insight": "While turnover has improved, it's still relatively high. Conduct exit interviews and employee surveys to identify and address remaining pain points."
    },
    {
        "query": "How has our social media engagement changed in the past month?",
        "response": "Our social media engagement (likes, shares, comments) has increased by 25% across all platforms.",
        "insight": "Strong growth in social media engagement. Analyze which content types are driving this increase and double down on those strategies."
    },
    {
        "query": "What percentage of our customers are using our mobile app?",
        "response": "Currently, 40% of our active customers are using our mobile app regularly.",
        "insight": "There's significant room for growth in mobile app adoption. Consider a targeted campaign to highlight app benefits and ease of use to non-users."
    },
    {
        "query": "How has our market share changed in the past year?",
        "response": "Our market share has grown from 15% to 17.5% over the past year.",
        "insight": "Steady market share growth indicates effective strategies. Analyze which product lines or regions contributed most to this growth for potential expansion opportunities."
    },
    {
        "query": "What's the average time to close a sale from initial contact?",
        "response": "The average time to close a sale is currently 45 days, down from 60 days last year.",
        "insight": "Significant improvement in sales cycle efficiency. Investigate which changes in the sales process led to this reduction and consider applying similar strategies across all sales teams."
    },
    {
        "query": "How many new features were implemented in our software this quarter?",
        "response": "We implemented 12 new features this quarter, compared to 8 in the previous quarter.",
        "insight": "Increased feature implementation rate. Ensure quality assurance keeps pace with this acceleration and gather user feedback to prioritize future development."
    },
    {
        "query": "What's our current ratio of repeat to new customers?",
        "response": "Our current ratio is 70% repeat customers to 30% new customers.",
        "insight": "Strong customer retention, but potential for growth in new customer acquisition. Review marketing strategies to ensure a balanced approach between retention and acquisition efforts."
    },
    {
        "query": "How has our website's bounce rate changed since the redesign?",
        "response": "The bounce rate has decreased from 55% to 40% since the website redesign.",
        "insight": "Significant improvement in user engagement post-redesign. Analyze which specific changes had the most impact and consider applying similar principles to other digital properties."
    },
    {
        "query": "What percentage of support tickets are resolved on first contact?",
        "response": "65% of support tickets are now resolved on first contact, up from 50% last quarter.",
        "insight": "Marked improvement in first-contact resolution rate. Investigate the factors contributing to this increase (e.g., better training, improved knowledge base) and reinforce these practices."
    }
]

# Function to validate JSON format
def is_valid_json(text):
    try:
        examples = json.loads(text)
        # Check if it's a list and has the required keys
        if not isinstance(examples, list):
            return False
        for example in examples:
            if not all(key in example for key in ["query", "response", "insight"]):
                return False
        return True
    except:
        return False

# Function to parse custom examples
def parse_custom_examples(text):
    try:
        return json.loads(text)
    except:
        return initial_examples




# Function to generate insights from queries and responses
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
    result = llm.invoke(final_prompt)
    return result.content.strip()

def generate_sql_query(df, natural_query):
    schema_info = "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns])
    
    prompt = f"""
    Given the following database schema and natural language query, generate a SQL query.
    The table name to use is 'df' (the data is in a pandas DataFrame).
    
    Table Schema:
    {schema_info}
    
    Sample data (first 5 rows):
    {df.head().to_string()}
    
    Natural Language Query: {natural_query}
    
    Return only the SQL query without any explanation. Use 'df' as the table name.
    """
    
    result = llm.invoke(prompt)
    sql_query = result.content.strip()
    sql_query = sql_query.replace('your_table_name', 'df')
    sql_query = sql_query.replace('your_table_name', 'df')
    return sql_query

def extract_numerical_data(response):
    """Extract numerical data from the response text"""
    # Look for percentages
    percentages = re.findall(r'(\d+(?:\.\d+)?)\%', response)
    # Look for numbers
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
    return [float(x) for x in percentages + numbers]

def create_visualization(df, data_type="numeric"):
    """Create appropriate visualization based on data type"""
    if data_type == "numeric":
        if len(df.columns) >= 2:
            fig = px.line(df, 
                         x=df.columns[0], 
                         y=df.columns[1:],
                         title='Data Visualization')
            fig.update_traces(mode='lines+markers')
        else:
            fig = px.bar(df, 
                        x=df.index, 
                        y=df.columns[0],
                        title='Data Visualization')
    else:
        # Create a pie chart for categorical data
        fig = px.pie(df, 
                    values=df.columns[1] if len(df.columns) > 1 else df.columns[0], 
                    names=df.columns[0],
                    title='Data Distribution')
    
    fig.update_layout(
        showlegend=True,
        hovermode='x unified'
    )
    return fig

def execute_sql_and_visualize(df, sql_query):
    try:
        env = {'df': df}
        result_df = sqldf(sql_query, env)
        
        if result_df.empty:
            st.warning("Query returned no results")
            return None, None
            
        # Determine if data is numeric or categorical
        data_type = "numeric" if result_df.select_dtypes(include=['float64', 'int64']).columns.any() else "categorical"
        fig = create_visualization(result_df, data_type)
        
        return result_df, fig
        
    except Exception as e:
        st.error(f"Error executing SQL query: {str(e)}")
        return None, None

def visualize_response(response):
    """Create visualization from natural language response"""
    numbers = extract_numerical_data(response)
    if numbers:
        df = pd.DataFrame({
            'Value': numbers,
            'Index': range(len(numbers))
        })
        fig = px.bar(df, x='Index', y='Value', title='Extracted Numerical Data')
        fig.update_layout(
            xaxis_title="Data Point",
            yaxis_title="Value"
        )
        return fig
    return None

# Improved function to query CSV data
def query_csv(df, query):
    # Get basic information about the DataFrame
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
    
    result = llm.invoke(prompt)
    return result.content.strip()


# Initialize the tracker in session state
if 'ai_tracker' not in st.session_state:
    st.session_state.ai_tracker = AIDecisionTracker()

# Modified Streamlit app
with col2:
    st.title("Chat with your CSV")

# Sidebar with custom examples option (keep existing sidebar code)
use_custom_examples = st.sidebar.checkbox("Use Custom Few-Shot Examples")
if use_custom_examples:
    # [Keep existing custom examples code]
    pass

# File uploader for CSV
with col2: 
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("CSV file uploaded successfully!")
        st.write(df.head())

        # User query input and insight generation
        user_query = st.text_input("Enter your business query about the CSV data:")

        if st.button("Generate Insights"):
            if user_query:
                with st.spinner("Analyzing CSV data and generating insights..."):
                    
                    # Generate SQL query
                    sql_query = generate_sql_query(df, user_query)
                    st.subheader("Generated SQL Query:")
                    st.code(sql_query, language='sql')
                    
                    # Allow query modification
                    modified_sql = st.text_area("You can modify the SQL query if needed:", sql_query)
                    
                    col1, col2 = st.columns(2)
                    

                    if st.button("Execute SQL Query"):
                        result_df, sql_fig = execute_sql_and_visualize(df, modified_sql)
                            
                        if result_df is not None:
                            st.subheader("SQL Query Results:")
                            st.write(result_df)
                                
                            if sql_fig is not None:
                                st.subheader("SQL Results Visualization:")
                                st.plotly_chart(sql_fig, use_container_width=True)
                                    
                                viz_type = st.selectbox(
                                        "Change visualization type:",
                                        ["Line", "Bar", "Scatter", "Area", "Pie"]
                                )
                                    
                                if st.button("Update Visualization"):
                                    if viz_type == "Line":
                                        new_fig = px.line(result_df, x=result_df.columns[0], y=result_df.columns[1:])
                                    elif viz_type == "Bar":
                                        new_fig = px.bar(result_df, x=result_df.columns[0], y=result_df.columns[1:])
                                    elif viz_type == "Scatter":
                                        new_fig = px.scatter(result_df, x=result_df.columns[0], y=result_df.columns[1:])
                                    elif viz_type == "Pie":
                                        new_fig = px.pie(result_df, values=result_df.columns[1], names=result_df.columns[0])
                                    else:  # Area
                                        new_fig = px.area(result_df, x=result_df.columns[0], y=result_df.columns[1:])
                                        
                                    new_fig.update_layout(
                                        title=f"Query Results - {viz_type} Chart",
                                        showlegend=True,
                                        hovermode='x unified'
                                        )
                                    st.plotly_chart(new_fig, use_container_width=True)
                    

                    # Natural language response and visualization
                    csv_response = query_csv(df, user_query)
                    st.subheader("Query Response:")
                    st.write(csv_response)
                        
                        # Create visualization for natural language response
                    response_fig = visualize_response(csv_response)
                    if response_fig is not None:
                        st.subheader("Response Data Visualization:")
                        st.plotly_chart(response_fig, use_container_width=True)
                            
                    # Generate insights
                    insight = analyze_responses(initial_examples, user_query, csv_response)
                    st.subheader("Generated Insight:")
                    st.write(insight)
                    

                    # Store current query info in session state
                    st.session_state.last_query = user_query
                    st.session_state.last_response = csv_response
                    st.session_state.last_insight = insight

                    # Add to chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "response": csv_response,
                        "insight": insight
                    })

        # AI Impact Tracking Section
        st.divider()
        st.header("AI Impact Tracking")
        with st.expander("Track Decision Impact"):
            metric_name = st.text_input("Metric Name (e.g., Sales, Customer Satisfaction)")
            baseline_value = st.number_input("Baseline Value", value=0.0)
            decision_made = st.text_area("Decision Made Based on AI Insight")
            predicted_impact = st.selectbox("Predicted Impact", ["Positive", "Negative", "Neutral"])
            
            if st.button("Track This Decision"):
                if hasattr(st.session_state, 'last_query'):
                    st.session_state.ai_tracker.add_decision(
                        st.session_state.last_query,
                        st.session_state.last_response,
                        st.session_state.last_insight,
                        decision_made,
                        predicted_impact,
                        metric_name,
                        baseline_value
                    )
                    st.success("Decision tracked successfully!")
                else:
                    st.warning("Please generate insights first before tracking a decision.")

        # Impact Update Section
        with st.expander("Update Impact Measurements"):
            if len(st.session_state.ai_tracker.tracking_df) > 0:
                decision_id = st.selectbox(
                    "Select Decision to Update",
                    options=st.session_state.ai_tracker.tracking_df.index.tolist()
                )
                new_value = st.number_input("Current Metric Value")
                if st.button("Update Impact"):
                    st.session_state.ai_tracker.update_impact(decision_id, new_value)
                    st.success("Impact updated successfully!")
            else:
                st.info("No decisions tracked yet.")

    # Enhanced Impact Dashboard with Visualizations
        st.header("Enhanced AI Impact Dashboard")
        
        # Summary metrics in cards
        impact_summary = st.session_state.ai_tracker.get_impact_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Decisions", impact_summary['total_decisions'])
        with col2:
            st.metric("Positive Impacts", impact_summary['positive_impacts'])
        with col3:
            success_rate = (impact_summary['positive_impacts'] / impact_summary['total_decisions'] * 100) if impact_summary['total_decisions'] > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col4:
            st.metric("Avg. Improvement", f"{impact_summary['average_improvement']:.2f}%")

        # Visualizations
        if len(st.session_state.ai_tracker.tracking_df) > 0:
            # Timeline Chart
            st.subheader("Decision Impact Timeline")
            timeline_chart = st.session_state.ai_tracker.create_timeline_chart()
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)

            # Two charts side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Impact Distribution")
                pie_chart = st.session_state.ai_tracker.create_impact_distribution_pie()
                if pie_chart:
                    st.plotly_chart(pie_chart, use_container_width=True)
            
            with col2:
                st.subheader("Metric Performance")
                metric_chart = st.session_state.ai_tracker.create_metric_performance_chart()
                if metric_chart:
                    st.plotly_chart(metric_chart, use_container_width=True)

            # Decision Activity Heatmap
            st.subheader("Decision Activity Over Time")
            heatmap = st.session_state.ai_tracker.create_decision_heatmap()
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)

            # Detailed Tracking Log with Filters
            st.subheader("Decision Tracking Log")
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                metric_filter = st.multiselect(
                    "Filter by Metric",
                    options=st.session_state.ai_tracker.tracking_df['metric_name'].unique()
                )
            with col2:
                impact_filter = st.multiselect(
                    "Filter by Impact",
                    options=st.session_state.ai_tracker.tracking_df['actual_impact'].unique()
                )
            with col3:
                date_range = st.date_input(
                    "Date Range",
                    value=(
                        st.session_state.ai_tracker.tracking_df['timestamp'].min().date(),
                        st.session_state.ai_tracker.tracking_df['timestamp'].max().date()
                    )
                )
            
            # Apply filters
            filtered_df = st.session_state.ai_tracker.tracking_df.copy()
            if metric_filter:
                filtered_df = filtered_df[filtered_df['metric_name'].isin(metric_filter)]
            if impact_filter:
                filtered_df = filtered_df[filtered_df['actual_impact'].isin(impact_filter)]
            if len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['timestamp'].dt.date >= date_range[0]) &
                    (filtered_df['timestamp'].dt.date <= date_range[1])
                ]
            
            st.dataframe(filtered_df)
            
            # Export functionality with filtered data
            if st.button("Export Filtered Data"):
                filtered_df.to_csv("ai_impact_tracking_filtered.csv")
                st.success("Filtered tracking data exported successfully!")
        else:
            st.info("Start tracking decisions to see visualizations and analytics.")
        

    # Keep existing instruction sidebar
    st.sidebar.markdown("""
    ## How to use:
    1. Upload a CSV file
    2. Enter your business query about the CSV data
    3. Click 'Generate Insights' to get AI-generated insights
    4. Track the impact of decisions made based on AI insights
    5. Update and monitor the impact of your decisions over time
    6. Export tracking data for further analysis
    """)

    # Keep existing DataFrame display options
    if uploaded_file is not None:
        if st.checkbox("Show full DataFrame"):
            st.write(df)
        
        if st.checkbox("Show DataFrame Statistics"):
            st.write(df.describe())
