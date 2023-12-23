import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, load_prompt
from langchain.chains import SequentialChain, LLMChain
from langchain.llms.openai import OpenAI

load_dotenv()

st.title("🦜🔗AI Blog writer Assistant")
st.sidebar.title("Give Your OpenAI API Key")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')


title_prompt_template = load_prompt("./prompts/title_prompt.yaml")
script_prompt_template = load_prompt("./prompts/script_prompt.yaml")

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    title_chain = LLMChain(llm=llm, prompt=title_prompt_template, output_key='title', verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_prompt_template, output_key='script', verbose=True)
    chain_seq = SequentialChain(
        chains=[title_chain, script_chain],
        input_variables=["topic"],
        output_variables=["title","script"],
        verbose=True
    )
    response = chain_seq({"topic":input_text})
    print(response)
    return response


with st.form('my_form'):
    topic = st.text_input('Enter your blog topic here')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        if topic:
            results = generate_response(topic)
            with st.expander(label="Title", expanded=False):
                st.write(results["title"])
            with st.expander(label="Script", expanded=False):
                st.write(results["script"])