from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
import re
import sqlalchemy
from sqlalchemy import create_engine

# Function to check for code blocks in the agent response and remove them from the displayed text
def check_for_code(agent_response):
    code_block_match = re.search(r'```(?:python)?(.*?)```', agent_response, re.DOTALL)
    if code_block_match:
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r'(?m)^\s*fig\.show\(\)\s*$', '', code_block)
        fig = get_fig_from_code(cleaned_code)
        response_text = re.sub(r'```(?:python)?(.*?)```', '', agent_response, flags=re.DOTALL).strip()
        return fig, response_text
    else:
        return None, agent_response

# Function to execute the extracted code and generate the figure
def get_fig_from_code(code):
    local_variables = {}
    exec(code, {}, local_variables)
    return local_variables.get('fig', None)

# Initialize the Streamlit app
st.set_page_config(page_title="ChatSQL", page_icon=":robot_face:")
st.header("ChatSQL")

# Sidebar for database details
st.sidebar.header("MySQL Database Connection")
db_host = st.sidebar.text_input("Host", value="localhost")
db_port = st.sidebar.text_input("Port", value="3306")
db_name = st.sidebar.text_input("Database Name")
db_user = st.sidebar.text_input("User")
db_password = st.sidebar.text_input("Password", type="password")
connect_button = st.sidebar.button("Connect")

# Session state initialization
if 'db' not in st.session_state:
    st.session_state.db = None
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'new_question' not in st.session_state:
    st.session_state.new_question = False

# Connect to the MySQL database when the button is clicked
if connect_button:
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        db = SQLDatabase(engine)
        st.session_state.db = db
        st.session_state.engine = engine
        st.success("Connection successful!")
    except sqlalchemy.exc.SQLAlchemyError as err:
        st.error(f"Error connecting: {err}")

# Chat interaction section
if st.session_state.db:
    llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0)
    agent_executor = create_sql_agent(llm, db=st.session_state.db, agent_type="openai-tools", verbose=True)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for question, response in st.session_state.chat_history:
            st.write(f"**Question:** {question}")
            if isinstance(response, tuple) and response[0]:
                st.plotly_chart(response[0])
            st.write(f"**Response:** {response[1]}")

    # Input section
    user_question = st.text_input("Describe a graph or Enter your question:")

    if user_question and not st.session_state.new_question:
        st.session_state.new_question = True
        with st.spinner(text="In progress..."):
            # Combine chat history with the new question
            prompt = """
            You are an expert data analyst, answer the next user query. The responses depend on the type of information requested in the query.
            If the query can be answered textually, provide the answer as a string based on the information in the database and the above chat history(if not blank).

            If the user asks for a graph or a visualization in his question, provide a description of the graph/visualization in the very beginning of the response and you must include the code to create the requested graph/visualization using plotly.(when creating the code, it should not be preceded by any introductions like 'here is the requested code...' or followed bydescriptions like 'this code generates the graph for...' etc.)

            If the answer is not known or available, respond with:
            "Requested information is not available in the database. Please try a different query."

            Return all output as a string.

            Now, let's tackle the query step by step. Here's the  user query for you to work on:
            """
            chat_history_text = "\n".join([f"Q: {q}\nA: {a[1]}" for q, a in st.session_state.chat_history])
            full_query = f"Chat history:{chat_history_text}\n {prompt}\n User Query: {user_question}"

            # Invoke the agent with the combined prompt and user question
            response = agent_executor.invoke(full_query)
            output = response.get('output', '')

            # Check if there is a code block in the output
            fig, response_text = check_for_code(output)
            st.session_state.chat_history.append((user_question, (fig, response_text)))

            # Rerun to update the chat history display
            st.rerun()

    st.session_state.new_question = False

# Closing the database connection
def close_db():
    if st.session_state.engine:
        st.session_state.engine.dispose()
        st.session_state.db = None
        st.session_state.engine = None
        st.success("Disconnected successfully!")

st.sidebar.button("Disconnect", on_click=close_db)
