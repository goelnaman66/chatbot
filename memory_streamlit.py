import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_astradb import AstraDBChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from astrapy import DataAPIClient
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType

st.set_page_config(page_title="Gemini Search Engine")
st.title("Search Engine")

with st.sidebar:
    st.title("Gemini Chatbot")
    api_key = st.text_input("Enter Google API Key", type="password")

os.environ["GOOGLE_API_KEY"] = api_key

# Astra DB configurations
astra_db_url = "https://5400c58a-0757-4cb7-8d8d-47a10a1ec55d-us-east-2.apps.astra.datastax.com"
astra_db_token = "AstraCS:wXrGKzIpDqQQQCoQpDPtxgAn:c7e8a29ab3b64f4e47b51aded8b4f56464c6f8d672b3a5612a99da1d9300185a"

# Getting all session_ids from the database
client = DataAPIClient(astra_db_token)
database = client.get_database(astra_db_url)

# Access collection and fetch unique session IDs
collection = database.get_collection("langchain_message_store")

# Fetch session IDs on initial load and store them in session state
if "session_ids" not in st.session_state:
    st.session_state.session_ids = collection.distinct("session_id")

# Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("Sessions")

# Set the default selected session if not already done
if "selected_session" not in st.session_state:
    st.session_state.selected_session = "None"  # Default to "None"

# Display session IDs as clickable buttons in the sidebar
selected_session = st.sidebar.radio(
    "Select a Session:",
    options=["None"] + st.session_state.session_ids,  # Add "None" as the default option
    index=0 if st.session_state.selected_session == "None" else st.session_state.session_ids.index(st.session_state.selected_session) + 1

)

# Update the session state based on the selected session
if selected_session is not None:
    st.session_state.selected_session = selected_session

# Divider for new session input
st.sidebar.divider()
st.sidebar.write("### Start a New Session")

# Input for a new session ID
new_session_id = st.sidebar.text_input("Enter a new Session ID:", value="")

# Add a button to create a new session
if st.sidebar.button("Create New Session"):
    if new_session_id and new_session_id not in st.session_state.session_ids:
        # Add the new session ID to the list
        st.session_state.session_ids.append(new_session_id)
        st.session_state.selected_session = new_session_id  # Set the new session as selected
        st.session_state.messages = []  # Clear messages for the new session
        st.success(f"New session '{new_session_id}' created successfully!")
    elif not new_session_id:
        st.error("Please enter a valid Session ID.")
    else:
        st.warning("Session ID already exists.")

# Display the session status
if st.session_state.selected_session:
    st.write(f"Active Session ID: {st.session_state.selected_session}")
else:
    st.write("No session selected yet.")
    




# tools:

# Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Wiki Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Duck go search tool
search = DuckDuckGoSearchRun(name="Search")

# Python REPL tool for complex calculations
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)



# LOADING MESSAGES AFTER SELECTING A SESSION ID
if astra_db_token and astra_db_url and st.session_state.selected_session:
    message_history = AstraDBChatMessageHistory(
        session_id=st.session_state.selected_session,
        api_endpoint=astra_db_url,
        token=astra_db_token)

    messages = message_history.messages
    # Update session state messages with fetched messages
    st.session_state.messages = messages

    if messages:
        for message in messages:
            if message.type == "human":
                with st.chat_message("user"):
                    st.write(message.content)
            elif message.type == "ai":
                with st.chat_message("assistant"):
                    st.write(message.content)

# Handle user input and update messages
if st.session_state.selected_session and api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    # tools = [wiki, arxiv, search, repl_tool]
    # search_agent = initialize_agent(
    #     tools, 
    #     llm, 
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    #     handle_parsing_errors=True
    # )

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: AstraDBChatMessageHistory(
            session_id=session_id,
            api_endpoint=astra_db_url,
            token=astra_db_token,
        ),
        input_messages_key="question",
        history_messages_key="history",
    )
    config = {"configurable": {"session_id": st.session_state.selected_session}}

    if user_input:= st.chat_input(disabled=st.session_state.selected_session == "None" or not api_key):
        with st.chat_message("user"):
            st.write(user_input)
        
        # Store user input in session state messages
        st.session_state.messages.append({"role":"user","content":user_input})
        
        if st.session_state.messages[-1]["role"]!="assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chain_with_history.invoke({"question": user_input}, config=config)
                    print(response)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response.content:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            message = {"role":"assistant", "content":full_response}
            st.session_state.messages.append(message)
