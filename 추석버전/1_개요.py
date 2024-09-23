import streamlit as st
from PyPDF2 import PdfReader
import docx
import tiktoken
from loguru import logger
import time
import concurrent.futures
import pandas as pd
import json
import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import joblib
import pickle
from fpdf import FPDF
from io import BytesIO
import base64

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

# 페이지 설정: 화면을 가로로 넓게 사용
st.set_page_config(layout="wide")

from funcs import load_css, load_local_font
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, 'style.css')

# CSS 파일 로드 함수
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"CSS 파일을 찾을 수 없습니다: {file_name}")

# CSS 파일 로드
load_css(css_path)

# RAG 설명 섹션
st.markdown(
    """
    <h3 style='font-size: 30px; font-family: Pretendard;'>|RAG란?</h3>
    """,
    unsafe_allow_html=True
)

# 파일이 있는지 확인하고 이미지를 로드
image_path4 = os.path.join(current_dir, "images", "langchain.jpg")

# 이미지가 있을 경우 표시
if os.path.exists(image_path4):
    st.image(image_path4, caption="RAG system 흐름도.", use_column_width=True)

st.markdown("* * *")

col1, col2 = st.columns(2)

# RAG 구축 설명 표시
with col1:
    st.write('''RAG(Retrieval-Augmented Generation: 검색증강생성)는 대규모 언어 모델의 출력을 최적화하여 응답을 생성하기 전에
                학습 데이터 소스 외부의 신뢰할 수 있는 지식 베이스를 참조하도록 하는 프로세스입니다.''')
    
    st.write(
        '''RAG는 LLM이 응답을 생성하기 전에 신뢰할 수 있는 외부 지식 베이스를 참조하도록 하여, 
        최신 정보 및 사용자가 원하는 도메인 정보를 반영하여 답변합니다. 아울러 참조할 수 있는 문서를 명확하게 지정해 주어
        답변의 부정확성이나 환각(hallucination)을 줄일 수 있습니다.'''
    )

with col2:
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
col1, col2 = st.columns(2)

# 이미지 경로 설정
image_path2 = os.path.join(current_dir, "images", "올라마.jpg")
image_path3 = os.path.join(current_dir, "images", "lmstudio.jpg")

# 이미지 로드 확인 및 표시
with col1:
    if os.path.exists(image_path2):
        st.image(image_path2, caption="올라마", use_column_width=True)
        st.markdown("[올라마 바로가기](https://ollama.com/)")

with col2:
    if os.path.exists(image_path3):
        st.image(image_path3, caption="lmstudio.", use_column_width=True)
        st.markdown("[lmstudio 바로가기](https://lmstudio.ai/)")

st.markdown("* * *")
image_path4 = os.path.join(current_dir, "images", "model.jpg")
if os.path.exists(image_path4):
    st.image(image_path4, caption="여러가지 모델", use_column_width=True)


st.write('''저는 속도가 빠르고 편리한 lmstudio를 사용했습니다.''')

st.write('''원하는 모델을 다운로드 받습니다. 
저는 teddylee777님이 올려주신 
teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf 모델을 다운로드 받았습니다. ''')

# 기술 스택 표시
st.markdown(
    """
    <h3 style='font-size: 30px; font-family: Pretendard;'>|tech stack</h3>
    """,
    unsafe_allow_html=True
)

# 이미지 경로 설정
image_path = os.path.join(current_dir, "images", "techstack.jpg")

# 파일이 있는지 확인하고 이미지를 로드
if os.path.exists(image_path):
    # width 파라미터를 사용하여 이미지 크기 조정
    st.image(image_path, width=200)  # width 값을 적절히 조정하여 크기 설정

st.markdown("* * *")
