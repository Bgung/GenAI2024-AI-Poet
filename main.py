import os
import langchain

import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain

load_dotenv()

prompt_gen = PromptTemplate(
    template="당신은 한국의 저명한 프롬프트 엔지니어입니다. 주제에 알맞으면서 창의적이고 운율이 훌륭한 시를 작성하기 위한 프롬프트를 작성하세요. 다양한 프롬프트 엔지니어링 기법을 사용해도 좋습니다. 주제는 '{subject}'입니다.",
    input_variables=["subject"]
)
poem_prompt = PromptTemplate(
    template="당신은 한국의 훌륭한 시인입니다. 아래에 맞게 훌륭한 시를 적절한 길이로 작성하세요. \n {prompt}",
    input_variables=["prompt"],
    output_variables=["poem"]
)
title_prompt = PromptTemplate(
    template="당신은 훌륭한 한국인 시인입니다. 아래 시에 대한 제목을 지어주세요.\n\n{poem}",
    input_variables=["poem"],
    output_variables=["title"]
)
tester_prompt = PromptTemplate(
    template="당신은 훌륭한 한국인 한국 시 평론가입니다. 아래 시에 대한 평가를 작성하세요.\n\n{poem}",
    input_variables=["poem"],
    output_variables=["evaluation"]
)


openai = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)

prompt_generator = LLMChain(
    llm=openai,
    prompt=prompt_gen,
    output_key="prompt"
)
poem_generator = LLMChain(
    llm=openai,
    prompt=poem_prompt,
    output_key="poem"
)
title_generator = LLMChain(
    llm=openai,
    prompt=title_prompt,
    output_key="title"
)
tester_chain = LLMChain(
    llm=openai,
    prompt=tester_prompt,
    output_key="evaluation"
)

chain = SequentialChain(
    chains=[prompt_generator, poem_generator, title_generator, tester_chain],
    input_variables=["subject"],
    output_variables=["prompt", "title", "poem", "evaluation"],
    return_all=True
)

st.title("인공지능 시인")
subject = st.text_input("시의 주제를 입력하세요")
st.write("시의 주제: ", subject)

if st.button("시 작성"):
    st.write("시 작성 중...")
    poem = chain.invoke({"subject": subject})
    print("Prompt: ", poem["prompt"])
    st.write("시 작성 완료")
    st.write("시 제목: ", poem["title"])
    st.write("시:\n", poem["poem"])
    st.write("평가: ", poem["evaluation"])
    