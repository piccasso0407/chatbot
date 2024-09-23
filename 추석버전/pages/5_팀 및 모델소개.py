import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from string import Template
import time
import streamlit as st
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os
# 페이지 설정: 화면을 가로로 넓게 사용
st.set_page_config(layout="wide")
# 페이지 로드
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
current_dir = os.path.dirname(os.path.abspath(__file__))
# 첫 번째 화면은 항상 표시되도록 설정
a, b = st.columns([2, 7])
with a:
    image_path = os.path.join(current_dir, "images", "hypre.jpg")
    if os.path.exists(image_path):
        st.image(image_path)
with b:
    st.markdown('####')
    team_title = '<b style="color:#31333f; font-size: 30px;">Team 고혈압</b>'
    st.markdown(team_title, unsafe_allow_html=True)
    st.write(
        """
        국민 건강 영양조사 원시데이터 자료를 분석하여 신체 데이터 및 설문 기반으로 여러 유사점을 찾고 분석한 이후 
        선별된 요인들을 토대로 고혈압을 예측할 수 있는 모델을 만들고 설문 조사를 통해 고혈압 확률뿐만 아니라 
        다양한 건강보고서를 통해 현재 자신의 신체 정보를 확인하며 여러 성인병을 예방할수 있는 방법을 제시하는 것이 
        이번 프로젝트의 궁극적인 주제와 목적입니다.         
        """
    )
    st.write(
        """
        건강한 신체를 유지하며 성인병의 주된 원인이라고 자주 지목되는 고혈압을 
        “먼저 예방하는 것이 다양한 합병증을 예방할 수 있겠다” 라는 가설을 세우며 고혈압을 예방할 수 있는 모델을 만들게 되었습니다. 
        또한 자신의 생활 습관에 관한 설문과 자신이 알고 있는 신체 데이터를 통해서 고혈압 확률을 예측할 수 있도록 
        모델을 설계하여 편리함과 정확성 둘 다 잡기 위해 노력했습니다.       
        """
    )
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



st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
        <a href="https://pressureproject.streamlit.app" target="_blank" style="text-decoration: none;">
            <div style="display: inline-block; padding: 10px 20px; background-color: rgb(83, 100, 147); color: white; border-radius: 5px; text-align: center; font-weight: bold;">
                고혈압예측모델보러가기
            </div>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
