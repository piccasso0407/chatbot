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

st.title('팀원소개')
a, b, c, d = st.columns([2, 2, 2, 2])
with a:
   st.markdown('### 김현빈')
   st.markdown('##### 프로젝트를 설계하는 데 있어,모든 작업 진행')
   st.write(''' 
            모델에 필요한 데이터를 다시 분석하기도 하고 다양한 모델 그리고 기법들을 활용하여 
            지속적인 모니터링을 통해 서비스에 사용될 최종 모델을 만들었습니다. 
            또한 이를 근거로 제시할 수 있는 Streamlit을 이용하여 누구든지 이해가 가능한 수준의 방식으로 페이지를 구성하였습니다.
            추가로 설문을 받아서 본인이 동나이 및 성별 대 위치를 제시하여 고혈압에 얼마나 노출되어있는지 알 수 있도록 구현하였습니다. 
            
            ''')
with b:
   st.markdown('### 신상길')
   st.markdown('##### 데이터 초기 이해와 분석과 데이터를 크롤링 및 검색')
   st.write(''' 
            다양한 피처 엔지니어링 방법을 공부하였고, 국민 건강 영양조사 데이터와 저희 팀의 방향성에 맞는 
            여러 머신러닝 모델을 돌려보았으며, 가설을 세워서 나름대로 모델 을 구현 해보기도했습니다.
            그 뒤로는 데이터 전처리 이전 부분을 스트림릿으로 구현해보았습니다. 
            지금은 정화님이 기틀을 잡은 코드를 기반으로 OpenAI API를 통해 챗봇을 구현하고 있습니다. 
            ''')
with c:
   st.markdown('### 이정화')
   st.markdown('##### 데이터 전처리와 및 모델링과 챗봇 구현 ')
   st.write('''
            전처리 과정에서 컬럼 선택과 모델링에 적극적으로 참여하였습니다. 
            3차 프로젝트에서는 챗봇구현을 담당하였는데 먼저 주차별 식단 데이터를 정리하고 
            텍스트 청크로 분리하여 분석에 용이하도록 만들었습니다. 
            이를 기반으로 임베딩과 벡터 스토어를 활용한 검색 시스템을 구축하여 
            사용자가 입력한 질의에 따라 맞춤형 식단을 추천하는 챗봇을 구현했습니다. 
            ''')
with d:
   st.markdown('### 정다운')
   st.markdown('##### 데이터 시각화와 논리적 통계 담당')
   st.write(''' 
            다양한 시각화를 통해 고혈압과 관련된 요인들을 찾아내고, 고혈압과 관련된 논문들을 읽어보고 
            여러 근거들을 통해 시각화의 기틀을 만들었습니다. 
            Streamlit을 사용하여 우리나라 고혈압 발병에 대한 통계와 
            프로젝트를 이해하는 데 있어 설명해주는 페이지를 만들었습니다. 
            ''')


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

st.write('''저는 두 개를 모두 사용하면서 비교해 보았는데 속도가 더 빠르고 편리하게 느껴졌던 lmstudio를 최종 선택했습니다.''')

st.markdown("* * *")

col1, col2 = st.columns(2)

with col1:
    st.write('''원하는 모델을 다운로드 받습니다. 
    저는 teddylee777님이 올려주신 
    teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf 모델을 다운로드 받았습니다. ''')
    

with col2:
    image_path5 = os.path.join(current_dir, "images", "model.jpg")
    if os.path.exists(image_path5):
        st.image(image_path5, caption="여러가지 모델", use_column_width=True)

# 기술 스택 표시

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

