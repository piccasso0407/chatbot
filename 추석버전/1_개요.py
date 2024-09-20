import streamlit as st
from PyPDF2 import PdfReader
import docx
import streamlit as st
import tiktoken
from loguru import logger
import time
import concurrent.futures
import pandas as pd
import json
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI




import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import joblib
import pickle
import os
from fpdf import FPDF
from io import BytesIO
import json

# funcs.py
from funcs import load_css, load_local_font
# 페이지 로드

# 페이지 설정
with st.sidebar:
    st.subheader("메뉴")
    st.markdown("* * *")
st.markdown("* * *")

# GitHub에서 CSS 파일 불러오기
github_url = "https://raw.githubusercontent.com/piccasso0407/chatbot/main/추석버전/style.css"  # 실제 URL로 변경

# CSS 파일 적용
def load_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

# CSS 파일 로드
load_css(github_url)

with st.sidebar:
    st.subheader("메뉴")
    st.markdown("* * *")
st.markdown("* * *")

st.markdown(
    """
    <h3 style='font-size: 30px; font-family: Pretendard;'>|tech stack</h3>
    """, 
    unsafe_allow_html=True
)

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "images", "techstack.jpg")

# 파일이 있는지 확인하고 이미지를 로드
if os.path.exists(image_path):
    st.image(image_path)


st.markdown("* * *")

current_subheader = "|RAG 구축"

# 만드는 방법
st.markdown(
    """
    <h3 style='font-size: 30px; font-family: Pretendard;'>|RAG란?</h3>
    """, 
    unsafe_allow_html=True
)
st.write('''RAG(Retrieval-Augmented Generation:검색증강생성)는 대규모 언어 모델의 출력을 최적화하여 응답을 생성하기 전에
            학습 데이터 소스 외부의 신뢰할 수 있는 지식 베이스를 참조하도록 하는 프로세스입니다.''')

st.write(
    '''RAG는 LLM이 응답을 생성하기 전에 신뢰할 수 있는 외부 지식 베이스를 참조하도록 하여, 
    최신 정보 및 사용자가 원하는 도메인 정보를 반영하여 답변합니다. 아울러 참조할 수 있는 문서를 명확하게 지정해 주어
    답변의 부정확성이나 환각(hallucination)을 줄일 수 있습니다.'''
)

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path4 = os.path.join(current_dir, "images", "langchain.jpg")

# 파일이 있는지 확인하고 이미지를 로드
if os.path.exists(image_path):
    st.image(image_path4, caption="RAG system 흐름도.", use_column_width=True)

st.write(
    '''
    1. RAG 모델은 다음과 같이 동작합니다.
    - 질문을 입력받습니다.
    - 질문을 임베딩하여 벡터로 표현합니다.
    - 사전에 벡터저장소에 저장된 문서 벡터들과 질문 벡터 간의 유사도를 계산합니다.
    - 유사도가 높은 상위 k개의 문서를 검색합니다.
    - 검색된 관련 문장들과 원래 질문을 템플릿에 삽입하여 프롬프트를 완성합니다.
    - 프롬프트를 LLM에 넣어 최종 답변을 생성합니다.
    '''
)
st.markdown("* * *")

# 기본설정
st.markdown(
    """
    <h3 style='font-size: 30px; font-family: Pretendard;'>|기본설정</h3>
    """, 
    unsafe_allow_html=True
)
col1, col2= st.columns(2)

image_path2 = os.path.join(current_dir, "images", "올라마.jpg")
image_path3 = os.path.join(current_dir, "images", "lmstudio.jpg")

# 파일이 있는지 확인하고 이미지를 로드

with col1:
    if os.path.exists(image_path):
        st.image(image_path2, caption="올라마", use_column_width=True)
        st.markdown("[올라마 바로가기](https://ollama.com/)")
with col2:
    if os.path.exists(image_path):
        st.image(image_path3,  caption="lmstudio.", use_column_width=True)
        st.markdown("[lmsutdio 바로가기](https://lmstudio.ai/)")

st.markdown("* * *")



    

