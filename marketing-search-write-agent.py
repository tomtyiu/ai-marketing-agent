import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
import logging
from functools import partial

from langchain_openai import ChatOpenAI
import asyncio  # Added for asynchronous processing

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Configuration and Initialization
# ------------------------------

# Set the page configuration
st.set_page_config(
    page_title="AI Marketing Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("AI Marketing Agent")

# Sidebar for configuration and example queries
st.sidebar.header("Configuration")

# API Key Input
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    help="Your API key is used to authenticate with OpenAI's services.",
)

# Avoid setting environment variables for better security; pass API key directly
# Removed: os.environ["OPENAI_API_KEY"] = api_key  # Removed to prevent side effects

# Check if the API key is available
if not api_key and not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# API Key Input
api_key2 = st.sidebar.text_input(
    "Enter your Tavily API Key:",
    type="password",
    help="Your API key is used to authenticate with OpenAI's services.",
)

# Check if the API key is available
if not api_key2 and not os.environ.get("TAVILY_API_KEY"):
    st.sidebar.warning("Please enter your Tavily API key to proceed.")
    st.stop()

# Sidebar for Configuration Parameters
st.sidebar.header("Parameters")

# Make 'max_results' configurable via the sidebar
max_results = st.sidebar.number_input(
    "Max Search Results",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Maximum number of search results to retrieve.",
)

# Example queries
st.sidebar.header("Example Queries")
example_queries = [
    "Who are the top five competitors in the AI agent market, and what are their market shares?",
    "What are the key strengths and weaknesses of Meta AI compared to our company?",
    "What are the current customer demographics and buying behaviors in the Artificial intelligence?",
    "How has our marketing ROI changed over the past 12 months across different campaigns?",
    "Identify the most effective digital marketing strategies for increasing lead generation in the SaaS (Software as a Service) sector targeting small businesses.",
    "Conduct a SWOT analysis for our latest marketing campaign and suggest areas for improvement.",
    "What are the top digital marketing trends predicted for 2025 in the wearable technology industry?",
    "How is the rise of artificial intelligence impacting marketing strategies in the online retail sector, specifically for personalized shopping experiences?",
    "What shifts are occurring in social media usage among our target audience, and how should we adapt?",
    "Compare the effectiveness of content marketing versus paid advertising in driving conversions for our new line of smart home devices."
]

# ------------------------------
# Initialize Session State
# ------------------------------
# Initialize session state with default values if not already present
st.session_state.setdefault("messages", [])
st.session_state.setdefault("user_query", "")
st.session_state.setdefault("agent_search", None)
st.session_state.setdefault("agent_writer", None)

# ------------------------------
# Agent Initialization with Caching
# ------------------------------
@st.cache_resource(show_spinner=False)
def initialize_agents(api_key, max_results):
    """
    Initializes and caches the search and writing agents to improve performance.
    """
    system_prompt_search = """
    You are an Marketing Agent with the capability to:
    1. Gather relevant literature from credible sources.
    2. Extract key findings and create advance concise summaries.
    3. Conduct competitive and market analysis and identify market trends in the data.
    4. Generate annotated bibliographies highlighting critical information about each source.
       1. Examine market data, consumer behavior, and competitor strategies to spot opportunities and potential challenges.
       2. Identify emerging market trends and articulate their significance for our marketing approach.
    5. Provide citations and references for all sources used in the market analysis.
    Reflect on the significance of the findings and synthesize logical conclusions.
    Always strive for clarity, accuracy, and intellectual rigor in your responses, citing authoritative sources when relevant. Remain neutral, logical, and helpful; avoid bias, speculation, or inappropriate language. Promote understanding by explaining reasoning step by step, and maintain academic integrity by transparently acknowledging any uncertainties. Respect user privacy and adhere to ethical standards in all analyses, while assisting in the discovery and interpretation of scholarly information.
    """

    system_prompt_writer = "You are an expert AI marketing campaign strategist with extensive experience in developing comprehensive and effective marketing campaigns across various industries. Your task is to create a detailed and professional marketing campaign."

    # Initialize memory and model with the provided API key
    memory = MemorySaver()
    model = ChatOpenAI(model="gpt-4o", api_key=api_key)  # Corrected model name and passed API key

    # Initialize search tool with user-configurable parameters
    search = TavilySearchResults(max_results=max_results,
                                 search_depth="advanced")  # Now configurable
    tools = [search]

    # Create agents with the respective system prompts
    agent_executor_search = create_react_agent(model, tools, state_modifier=system_prompt_search)
    agent_executor_writer = create_react_agent(model, tools, state_modifier=system_prompt_writer)

    return agent_executor_search, agent_executor_writer

async def async_initialize_agents(api_key, max_results):
    """
    Asynchronous wrapper for initializing agents to prevent blocking.
    """
    loop = asyncio.get_event_loop()
    agent_search, agent_writer = await loop.run_in_executor(
        None, partial(initialize_agents, api_key, max_results)
    )
    return agent_search, agent_writer

# Initialize agents and store them in session state
if st.session_state.agent_search is None or st.session_state.agent_writer is None:
    with st.spinner("Initializing AI agents..."):
        try:
            agent_search, agent_writer = initialize_agents(api_key or os.environ["OPENAI_API_KEY"], max_results)
            st.session_state.agent_search = agent_search
            st.session_state.agent_writer = agent_writer
            logger.info("Agents initialized successfully.")
        except Exception as e:
            st.error(f"Failed to initialize agents: {e}")
            logger.error(f"Agent initialization error: {e}")
            st.stop()

# ------------------------------
# Agent Function Definition
# ------------------------------
def search_write_agent(objective_text, agent_search, agent_writer):
    """
    Creates and invokes two agents:
    1. Search Agent: Gathers relevant literature based on the user's query.
    2. Writing Agent: Generates a professional document based on the search results.

    Returns the responses from both agents.
    """
    logger.info(f"Received objective_text: {objective_text}")

    # Invoke the search agent
    try:
        search_response = agent_search.invoke({"messages": [HumanMessage(content=objective_text)]})
        logger.info("Search agent invoked successfully.")
    except Exception as e:
        logger.error(f"Error invoking search agent: {e}")
        raise e

    # Extract and format search responses using list comprehension for efficiency
    search_messages = [
        f"{message.__class__.__name__}: {message.content}"
        for message in search_response.get('messages', [])
    ]

    logger.info(f"Search messages: {search_messages}")

    # Combine search messages into a single string for the writer agent
    combined_search_content = "\n\n".join(search_messages)

    # Invoke the writing agent with the search results
    try:
        write_response = agent_writer.invoke({"messages": [HumanMessage(content=combined_search_content)]})
        logger.info("Writing agent invoked successfully.")
    except Exception as e:
        logger.error(f"Error invoking writing agent: {e}")
        raise e

    # Extract and format writing responses using list comprehension
    write_messages = [
        f"{message.__class__.__name__}: {message.content}"
        for message in write_response.get('messages', [])
    ]

    logger.info(f"Writing messages: {write_messages}")

    return search_messages, write_messages

# ------------------------------
# Chat Display Function
# ------------------------------
def display_chat():
    """
    Displays the chat messages stored in session state.
    """
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(f"**You:** {msg['content']}")
        elif msg['role'] == 'assistant_search':
            with st.chat_message("assistant"):
                st.markdown("**Search Agent:**")
                st.markdown(msg['content'])
        elif msg['role'] == 'assistant_writer':
            with st.chat_message("assistant"):
                st.markdown("**Writing Agent:**")
                st.markdown(msg['content'])

# ------------------------------
# Submit Function Definition
# ------------------------------
def submit():
    """
    Handles the submission of a user query.
    """
    user_input = st.session_state.user_query.strip()  # Remove leading/trailing whitespace
    if user_input:
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
    else:
        # Handle cases with no query input
        st.session_state.messages.append({"role": "user", "content": "[No query provided]"})

    display_chat()

    # Proceed only if there is a query to process
    if user_input:
        # Run the agents
        with st.spinner("Processing your request..."):
            try:
                # Consider making this an asynchronous call if the initialize_agents is also async
                search_msgs, write_msgs = search_write_agent(
                    user_input,
                    st.session_state.agent_search,
                    st.session_state.agent_writer
                )
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                logger.error(f"Processing error: {e}")
                return

        # Append search agent messages to chat history
        for msg in search_msgs:
            st.session_state.messages.append({"role": "assistant_search", "content": msg})

        # Append writing agent messages to chat history
        for msg in write_msgs:
            st.session_state.messages.append({"role": "assistant_writer", "content": msg})

    # Clear the input box
    # st.session_state.user_query = ""

    # Refresh the chat display
    display_chat()

# ------------------------------
# Callback Function for Example Queries
# ------------------------------
def example_query_callback(query):
    st.session_state.user_query = query
    submit()

# ------------------------------
# Chat Interface Display
# ------------------------------
display_chat()

# ------------------------------
# Example Query Buttons
# ------------------------------
for query in example_queries:
    if st.sidebar.button(query):
        example_query_callback(query)

# ------------------------------
# User Input Form
# ------------------------------
with st.form(key='input_form', clear_on_submit=True):
    user_query = st.text_input("Enter your query for search and write:", key="user_query")
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        submit()

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Developed with using Streamlit and LangChain.")

