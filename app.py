import streamlit as st
from openai import OpenAI
import os
import sys
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv, dotenv_values
load_dotenv()


if 'key' not in st.session_state:
    st.session_state['key'] = 'value'




# initialize the client but point it to TGI
client = OpenAI(
  base_url="https://api-inference.huggingface.co/v1",
  api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')#"hf_xxx" # Replace with your token
) 




#Create supported models
model_links ={
    "Mistral":"mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma":"google/gemma-7b-it",
    "Llama-2":"meta-llama/Llama-2-7b-chat-hf"

}



# Define the available models
# models = ["Mistral", "Gemma"]
models =[key for key in model_links.keys()]

# Create the sidebar with the dropdown for model selection
selected_model = st.sidebar.selectbox("Select Model", models)



#Pull in the model we want to use
repo_id = model_links[selected_model]



st.title(f'ChatBot Using {selected_model}')

# Set a default model
if selected_model not in st.session_state:
    st.session_state[selected_model] = model_links[selected_model] 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Accept user input
if prompt := st.chat_input("What is up?"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())

        stream = client.chat.completions.create(
            model=model_links[selected_model],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=0.5,
            stream=True,
            max_tokens=3000,
        )

        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})