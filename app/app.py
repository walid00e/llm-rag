import streamlit as st
import requests

add_title = st.sidebar.title("Configure LLM")
# Add a selectbox to the sidebar:
model = st.sidebar.selectbox(
    'Specify the model:',
    ('llama3.2:latest', 'gpt-oss:latest', 'deepseek-r1:latest', 'gemma3:latest')
)
with st.sidebar.expander("Advanced Settings", expanded=False):
    reasoning = st.checkbox("Use Reasoning")
    topK = st.slider(
        'Top-k:',
        0, 100, 100
    )
    topP = st.slider(
        'Top-p:',
        0.0, 1.0, 1.0
    )
    temperature = st.slider(
        'Temperature:',
        0.0, 2.0, 1.0
    )
    num_predict = st.slider(
        'num predict:',
        512, 4096, 512
    )

with st.sidebar.expander("Added Instructions", expanded=False):
    instructions = st.text_area(
        "Enter your instructions:",
        placeholder="Type your instructions here...",
        height=150
    )

if st.sidebar.button("Update", type="primary", width="stretch"):
    payload = {
        "model": model,
        "reasoning": reasoning,
        "temperature": temperature,
        "top_k": topK,
        "top_p": topP,
        "instructions": instructions,
        "numPredict":num_predict
    }
    try:
        response = requests.post("http://localhost:8000/config_llm", json=payload)
        if response.status_code == 200:
            st.sidebar.success("LLM configuration updated!")
        else:
            st.sidebar.error(f"Failed to update LLM: {response.text}")
    except Exception as e:
        st.sidebar.error(f"Error connecting to server: {e}")

with st.sidebar.expander("Upload Document", expanded=False):
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf"],
        help="Upload a document to add to the knowledge base",
        key = f"uploader_{st.session_state.uploader_key}"
    )
    upload_button = st.button("Upload Document", use_container_width=True)
    if upload_button:
        if uploaded_file is not None:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(
                    "http://localhost:8000/upload_document",  # Your upload endpoint
                    files=files
                )
                if response.status_code == 200:
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    st.sidebar.error(f"Upload failed")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.warning("Please select a file first")

with st.sidebar.expander("Documents List", expanded=False):
    response = requests.get("http://localhost:8000/get_documents")
    if response.status_code == 200:
        docs = response.json().get("documents", [])
        for doc in docs:
            st.text(f"📕 {doc.split("_")[1]}")

with st.sidebar.expander("Chat Controls"):
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

API_URL = "http://localhost:8000/ask"  # your endpoint

def call_backend(query: str) -> str:
    """Send the user input to the backend and return the textual response."""
    payload = {"prompt": query}
    try:
        r = requests.post(API_URL, json=payload, timeout=200)
        r.raise_for_status()
        data = r.json()
        return data.get("answer", "No answer field returned.")
    except Exception as e:
        return f"Backend error: {e}"

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
user_text = st.chat_input("What is up?")

if user_text:
    # Register user request
    st.chat_message("user").markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Query your backend
    backend_answer = call_backend(user_text)

    # Display backend response
    with st.chat_message("assistant"):
        st.markdown(backend_answer)
    st.session_state.messages.append(
        {"role": "assistant", "content": backend_answer}
    )