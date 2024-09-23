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
        RAG 모델은 다음과 같이 동작합니다.
        - 질문을 입력받습니다.
        - 질문을 임베딩하여 벡터로 표현합니다.
        - 사전에 벡터저장소에 저장된 문서 벡터들과 질문 벡터 간의 유사도를 계산합니다.
        - 유사도가 높은 상위 k개의 문서를 검색합니다.
        - 검색된 관련 문장들과 원래 질문을 템플릿에 삽입하여 프롬프트를 완성합니다.
        - 프롬프트를 LLM에 넣어 최종 답변을 생성합니다.
        '''
    )
st.markdown("* * *")
st.subheader("|RAG의 장단점")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 장점")
    st.write("""
    - 최신 정보 반영 가능
    - 특정 도메인 지식 활용 가능
    - 환각(hallucination) 감소
    - 모델 재학습 없이 지식 확장 가능
    """)
with col2:
    st.markdown("#### 단점")
    st.write("""
    - 검색 품질에 따른 성능 의존성
    - 추가적인 계산 비용
    - 시스템 복잡도 증가
    """)

st.markdown("* * *")
st.subheader("|프로젝트 구조")
st.code("""
project/
│
├── pages/
│   ├── 1_개요.py
│   ├── 2_RAG구축.py
│   ├── 3_성능평가.py
│   └── 4_식단추천.py
│
├── data/
│   └── (원본 문서 파일들)
│
└── images/
    └── (사용된 이미지 파일들)

""")
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

st.write('''저는 두 개를 모두 사용하면서 비교해 보았는데 속도가 더 빠르고 편리하게 느껴졌던 lmstudio를 최종 선택했습니다.''')

st.markdown("* * *")
# 모델 선택 및 설정
st.subheader("|모델 선택 및 설정")
st.write("본 프로젝트에서는 다음 모델들을 테스트하였습니다:")
st.write("1. Meta-Llama-3.1-8B-Instruct-GGUF")
st.write("2. teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf")
st.markdown("* * *")

image_path = os.path.join(current_dir, "images", "techstack.jpg")

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "images", "techstack.jpg")

# 파일이 있는지 확인하고 이미지를 로드
if os.path.exists(image_path):
    # 이미지 파일을 base64로 인코딩하여 HTML로 삽입
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode()

    # HTML을 사용하여 이미지 가운데 정렬
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{img_base64}" alt="Image" style="width: 300px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")

st.write("""
모델 선택 시 고려사항:
- 한국어 성능
- 모델 크기 및 로컬 실행 가능성
- 추론 속도
- 메모리 요구사항
""")

# 다음 단계
st.subheader("|다음 단계")
st.write("""
이어지는 페이지에서는 다음 내용을 다룹니다:
1. 데이터 준비 및 전처리
2. 임베딩 생성 및 벡터 데이터베이스 구축
3. 질의응답 시스템 구현 및 성능테스트
""")
