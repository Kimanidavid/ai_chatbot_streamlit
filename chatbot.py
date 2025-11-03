import streamlit as st
from transformers import pipeline

# streamlit page setup
st.set_page_config(page_title="Chatbot", page_icon="🤖")

#load pretrained model
def load_text_generator():
    text_generator = pipeline("text-generation", model="gpt2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos_token
    return text_generator

# system behaviour prompt
SYSTEM_PROMPT = """You are a helpful assistant. 
                Answer the user's questions to the best of your ability.
                do not halucinate information.
                If you don't know the answer, just say that you don't know.
                you are to help the user with concise and accurate information.
                keep your answers brief and to the point."""

# build conversation prompt
def build_prompt(chat_history, user_input):
    formated_conversation = []
    for previous_question, previous_answer in chat_history:
        formated_conversation.append(f"User: {previous_question}\nAssistant: {previous_answer}")
    formated_conversation.append(f"User: {user_input}\nAssistant:")
    return SYSTEM_PROMPT + "\n".join(formated_conversation)

# page headers
st.title("🤖 David's Chatbot")

#building sidebar
with st.sidebar:
    st.title("Settings")
    max_new_tokens = st.slider("Max New Tokens", min_value=50, max_value=500, value=150, step=25)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

    if st.button("clear chat history"):
        st.session_state.chat_history = []

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# displaying chat history
for user_question, bot_answer in st.session_state.chat_history:
    st.chat_message(user_question, is_user=True).markdown(user_question)
    st.chat_message(bot_answer).markdown(bot_answer)

# user input field
user_input = st.chat_input("Ask me anything!")
if user_input:
    st.chat_message("user_input", is_user=True).markdown(user_input)

    with st.spinner("Thinking..."):
        # load model
        text_generator = load_text_generator()
        # build conversation prompt
        prompt_text = build_prompt(st.session_state.chat_history, user_input)

        generation_output = text_generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=text_generator.tokenizer.eos_token_id,
            do_sample=True,
        )[0]["generated_text"]

        # extracting ai answer
        bot_answer = generation_output.split("Assistant:")[-1].strip()
        if "User:" in bot_answer:
            bot_answer = bot_answer.split("User:")[0].strip()

    # display bot answer
    st.chat_message("bot_answer").markdown(bot_answer)
    # update chat history
    st.session_state.chat_history.append((user_input, bot_answer))
