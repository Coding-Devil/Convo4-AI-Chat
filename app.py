import streamlit as st
from openai import OpenAI
import os
import sys
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv, dotenv_values
load_dotenv()





# initialize the client but point it to TGI
client = OpenAI(
  base_url="https://api-inference.huggingface.co/v1",
  api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')#"hf_xxx" # Replace with your token
) 




#Create supported models
model_links ={
    "Mistral":"mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma":"google/gemma-7b-it",
    # "Llama-2":"meta-llama/Llama-2-7b-chat-hf"

}

#Pull info about the model to display
model_info ={
    "Mistral":
        {'description':"""The Mistral model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nIt was created by the [**Mistral AI**](https://mistral.ai/news/announcing-mistral-7b/) team as has over  **7 billion parameters.** \n""",
        'logo':'https://mistral.ai/images/logo_hubc88c4ece131b91c7cb753f40e9e1cc5_2589_256x0_resize_q97_h2_lanczos_3.webp'},
    "Gemma":        
        {'description':"""The Gemma model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nIt was created by the [**Google's AI Team**](https://blog.google/technology/developers/gemma-open-models/) team as has over  **7 billion parameters.** \n""",
        'logo':'https://pbs.twimg.com/media/GG3sJg7X0AEaNIq.jpg'},

}


# Define the available models
# models = ["Mistral", "Gemma"]
models =[key for key in model_links.keys()]

# Create the sidebar with the dropdown for model selection
selected_model = st.sidebar.selectbox("Select Model", models)

# Create model description
st.sidebar.write(f"You're now chatting with **{selected_model}**")
st.sidebar.markdown(model_info[selected_model]['description'])
st.sidebar.image(model_info[selected_model]['logo'])


#Pull in the model we want to use
repo_id = model_links[selected_model]


st.subheader(f'AI - {selected_model}')
# st.title(f'ChatBot Using {selected_model}')

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
if prompt := st.chat_input(f"Hi I'm {selected_model}, ask me a question"):

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