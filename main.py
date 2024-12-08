import os
import langchain

import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

st.title("인공지능 시인")
subject = st.text_input("시의 주제를 입력하세요")
st.write("시의 주제: ", subject)

if st.button("시 작성"):
    st.write("시 작성 중...")
    poem = openai.invoke(
        input=f"당신은 훌륭한 한국인 시인이고, 200자 정도의 길이를 가진 시를 쓰는 중입니다 주제에 알맞으면서 최대한 창의적이고, 운율에 맞게 작성하세요. 주제는 '{subject}'입니다.",
    )
    st.write(poem)