import streamlit as st
import os
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from string import Template
import re
import time
from langchain.prompts import ChatPromptTemplate
from funcs import load_css
import gc
import pandas as pd
import requests
import json
from langchain.memory import ConversationBufferMemory
import PyPDF2
import streamlit as st
import os


st.set_page_config(layout="wide")
# 페이지 로드
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

# Pretendard 폰트 로드
st.markdown("""
    <link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Pretendard', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)


# 현재 스크립트의 절대 경로를 얻습니다
current_dir = os.path.dirname(os.path.abspath(__file__))

# 동영상 파일의 절대 경로를 생성합니다
video_path = os.path.join(current_dir, "images", "bandicam 2024-09-27 08-50-24-796.mp4")

# 동영상 파일이 존재하는지 확인합니다
if os.path.exists(video_path):
    # 동영상 파일을 열고 읽습니다
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()

    # 동영상을 표시합니다
    st.video(video_bytes)
else:
    st.error("동영상 파일을 찾을 수 없습니다.")
