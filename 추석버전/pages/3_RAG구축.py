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


current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "images", "주차별메뉴.jpg", )
image_path2 = os.path.join(current_dir, "images", "벡터스토어.jpg")
image_path3 = os.path.join(current_dir, "images", "임베딩.jpg")
# 파일이 있는지 확인하고 이미지를 로드
st.write("제 RAG시스템의 기본 틀은 유튜브 채널 모두의 AI에서 참조했습니다.  
st.markdown("[모두의 AI 채널 바로가기](https://youtu.be/xYNYNKJVa4E?si=VCFXPu9vXZIMbjhh)")

st.markdown("* * *")
st.subheader("| 파일 읽기")
st.image(image_path)
# 파일 경로

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'week_menu_by_day.pdf')


# PDF 파일 읽기 함수
def read_pdf(file_path):
    pdf_text = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text.append(page.extract_text())
    return pdf_text

# PDF 파일에서 텍스트 불러오기
pdf_content = read_pdf(file_path)
len_str = str(len(pdf_content))

st.code(''' 
            def read_pdf(file_path):
    pdf_text = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text.append(page.extract_text())
    return pdf_text
            ''')
st.text_area("첫 번째 페이지의 내용:", pdf_content[0], height=300)
st.write(f"전체 페이지 수:{len_str}")


# 데이터를 문서 리스트로 변환
documents = [Document(page_content=text) for text in pdf_content]

st.markdown("* * *")
st.subheader("| 청크 나누기")
st.write("청크란?")
st.write("""rag를 구축하는 데 있어 시스템의 효율성과 성능을 결정하는 중요한 매개 변수 중 하나이다.
        임베딩에서 알고리즘도 중요하지만 가장 중요한 것중 하나는 어떻게 문서를 파편으로 잘라낼것인가? 이다. 
        임베딩은 텍스트를 고정된 크기의 벡터로 변경하는 것이기 때문에, 긴 문단을 작은 벡터로 임베딩하게 되면 
        디테일한 의미를 잃어버릴 수 있고, 반대로 작은 문장으로 임베딩을 하면,
        검색시 문장에 대한 정확도는 올라가겠지만, 문장이 짧아서 문장 자체가 가지고 있는 정보가 부족하게 된다.

        -출처: https://bcho.tistory.com/1404 [조대협의 블로그:티스토리]""")
st.markdown("""
    [LLM chunk size를 결정하는 방법](https://velog.io/@js03210/LLM-chunk-size%EB%A5%BC-%EA%B2%B0%EC%A0%95%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95)
    """)

st.code('''
        # 텍스트 청크 분리 함수
def get_text_chunks(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks''')

# 청크 사이즈 및 오버랩 설정
chunk_size = st.slider("청크 사이즈를 선택하세요:", min_value=100, max_value=5000, value=1000, step=100)
chunk_overlap = st.slider("청크 오버랩을 선택하세요:", min_value=0, max_value=5000, value=400, step=100)

# 텍스트 청크 분리 함수
def get_text_chunks(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if documents:
    # 슬라이더에서 설정한 chunk_size와 chunk_overlap을 함수에 전달
    text_chunks = get_text_chunks(documents, chunk_size, chunk_overlap)
    st.write(f"청크의 개수: {len(text_chunks)}")
    st.text_area("첫 번째 청크:", text_chunks[0].page_content if text_chunks else "청크가 없습니다.", height=300)
else:
    st.write("문서를 읽어오지 못했습니다.")


st.markdown("* * *")
st.subheader('| 벡터스토어')
st.write("여러가지 벡터스토어")
# 파일이 있는지 확인하고 이미지를 로드
st.image(image_path2)

st.markdown("#### faiss와 chroma의 차이")
st.markdown("""
    [ChromaDB vs FAISS 비교](https://medium.com/@sujathamudadla1213/chromadb-vsfaiss-65cdae3012ab)
""")




selected_model = "jhgan/ko-sroberta-multitask"


# 벡터 스토어 생성 함수
@st.cache_resource
def get_vectorstore(_text_chunks, selected_model):
    embeddings = HuggingFaceEmbeddings(
        model_name=selected_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(_text_chunks, embeddings)
    return vectordb
st.code('''
def get_vectorstore(_text_chunks, selected_model):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(_text_chunks, embeddings)
    return vectordb
''')

st.markdown("* * *")
st.subheader("| 임베딩 모델")
st.image(image_path3)
st.markdown(
    """
    [임베딩이란? (LLM 활용을 위한)](https://www.gnict.org/blog/130/%EA%B8%80/llm%ED%99%9C%EC%9A%A9%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%9E%84%EB%B2%A0%EB%94%A9%EC%9D%B4%EB%9E%80/)
    """,
    unsafe_allow_html=True)
# 사용자가 선택할 임베딩 모델 옵션

# 텍스트 청크 생성 및 벡터스토어 생성
if documents:
    text_chunks = get_text_chunks(documents, chunk_size, chunk_overlap)
    vectorstore = get_vectorstore(text_chunks, selected_model)
    st.write("벡터스토어가 생성되었습니다.")

# 벡터 검색 함수
def get_word_vector(query, selected_model):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    query_embedding = embeddings.embed_query(query)
    return query_embedding

query = st.text_input("벡터를 확인할 단어를 입력하세요:")

# 단어 벡터 추출 및 표시
if query:
    word_vector = get_word_vector(query, selected_model)

    # 벡터의 차원과 앞 5개의 값을 출력
    vector_dimension = len(word_vector)
    first_five_values = word_vector[:5]
    
    st.write(f"'{query}'의 벡터 차원: {vector_dimension}차원")
    st.write("벡터의 앞 5개 값:")
    for i, value in enumerate(first_five_values):
        st.write(value)

def calculate_cosine_similarity(vec1, vec2):
    # 코사인 유사도는 벡터 내적을 두 벡터의 노름(크기)으로 나눈 값
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity

query1 = query
query2 = st.text_input("두 번째 단어를 입력하세요:")

# 단어 벡터 추출 및 코사인 유사도 계산
if query1 and query2:
    word_vector1 = get_word_vector(query1, selected_model)
    word_vector2 = get_word_vector(query2, selected_model)
    
    # 두 벡터 간의 코사인 유사도 계산
    cosine_sim = calculate_cosine_similarity(word_vector1, word_vector2)
    
    # 결과 출력
    st.write(f"'{query1}'와 '{query2}' 간의 코사인 유사도: {cosine_sim:.4f}")

st.markdown("* * *")
st.subheader("| 대화형 체인")

# 대화형 체인 생성 함수 (히스토리 없이)
def get_conversation_chain(_vectorstore):
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
        temperature=0.1,
        streaming=True
    )

    retriever = _vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 5},
        verbose=False
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever, 
        return_source_documents=True,
        verbose=False
    )
    return conversation_chain



st.code('''
def get_conversation_chain(_vectorstore):
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="teddylee777/EEVE-Korean-Instruct-10.8B-v1.0-gguf",
        temperature=0.1,
        streaming=True
    )

    retriever = _vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 5},
        verbose=False
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever, 
        return_source_documents=True,
        verbose=False
    )
    return conversation_chain
''')
# 문서를 청크로 분리
text_chunks = get_text_chunks(documents, chunk_size, chunk_overlap)

# 벡터 스토어 생성
vectorstore = get_vectorstore(text_chunks, selected_model)

# 대화형 체인 생성
conversation_chain = get_conversation_chain(vectorstore)

st.markdown("* * *")
st.subheader("| 프롬프트엔지니어링")

st.code('''
   def get_prompt_template(query):
    pdf_content_str = "\n".join(pdf_content2) if isinstance(pdf_content2, list) else str(pdf_content2)
    
    if "식단" in query:
        prompt = f"""
        작업: 사용자의 건강 정보를 바탕으로 맞춤형 식단 추천

        참고 정보:
        1. 사용자 건강 정보: {pdf_content_str}
        2. 식단 출처: {docx_file_path} 파일의 내용만 사용할 것

        지침:
        1. 기간: 사용자가 특정 기간을 요청하면 그 기간에 맞춰, 아니면 하루 식단을 추천하세요.
        2. 형식: 
           아침: [메뉴1], [메뉴2]
           점심: [메뉴1], [메뉴2]
           저녁: [메뉴1], [메뉴2]
        3. 나트륨 제한:
           - 고혈압 확률 20% 초과: 나트륨 1000mg 이하 식단
           - 고혈압 확률 20% 미만: 나트륨 제한 없음
        4. 추가 정보:
           - 각 식단의 영양 정보를 포함해 주세요.
           - 모든 링크는 제거해 주세요.
           - 아침, 점심, 저녁은 각각 중복이 없게해 주세요.
 \
        응답 형식:
        1. 추천 식단 (위 형식대로)
        2. 각 식단의 영양 정보
        3. 식단 선택 이유 간단히 설명

        질문: {query}
        """
    elif "레시피" in query:
        prompt = f"""
        작업: 특정 요리의 레시피 설명

        참고 정보:
        1. 레시피 출처: {docx_file_path} 파일

        지침:
        1. {query}의 레시피를 자세히 설명해 주세요.
        2. 모든 링크는 제거해 주세요.
        3. 해당 요리의 영양 정보를 포함해 주세요.


        응답 형식:
        1. 재료 목록
        2. 조리 단계
        3. 영양 정보
        4. 조리 팁 (선택사항)

        질문: {query}
        """
    else:
        prompt = f"""
        작업: 일반적인 식단 및 건강 관련 질문 답변

        참고 정보:
        1. 사용자 건강 정보: {pdf_content_str}
        2. 추가 정보 출처: {docx_file_path} 파일

        지침:
        1. 제공된 정보를 바탕으로 질문에 자유롭게 답변해 주세요.
        2. 가능한 한 구체적이고 정확한 정보를 제공해 주세요.

        응답 형식:
        1. 질문에 대한 직접적인 답변
        2. 추가 설명 또는 관련 정보 (필요시)
        3. 주의사항 또는 권고사항 (적절한 경우)

        질문: {query}
        """
    return prompt''')
st.markdown("* * *")
st.subheader("| 대화체인 호출")

st.code('''
chain = conversation_chain
result = chain({"question": formatted_query, "chat_history": []})
response = result['answer']
source_documents = result['source_documents']
''')




