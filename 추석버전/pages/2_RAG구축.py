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

# 페이지 로드


current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "images", "주차별메뉴.jpg")
image_path2 = os.path.join(current_dir, "images", "벡터스토어.jpg")
image_path3 = os.path.join(current_dir, "images", "임베딩.jpg")
# 파일이 있는지 확인하고 이미지를 로드
st.subheader("1. 파일 읽기")
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


st.subheader("2. 청크 나누기")
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



st.subheader('3. 벡터스토어')
st.write("여러가지 벡터스토어")
st.image(image_path2)
st.write("faiss와 chroma의 차이")
st.markdown("""
    [ChromaDB vs FAISS 비교](https://medium.com/@sujathamudadla1213/chromadb-vsfaiss-65cdae3012ab)
""")




selected_model = "jhgan/ko-sroberta-multitask"


# 벡터 스토어 생성 함수
@st.cache_resource
def get_vectorstore(_text_chunks, selected_model):
    embeddings = HuggingFaceEmbeddings(
        model_name=selected_model,
        model_kwargs={'device': 'cuda'},
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
st.subheader("4. 임베딩 모델이란?")
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


st.subheader("5. 대화형 체인")

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

st.subheader("6. 프롬프트엔지니어링")

st.code('''
    def get_prompt_template(query):
    # "고혈압"이 포함된 경우 나트륨 1000 이하 식단 추천
    if "고혈압" in query:
        prompt = Template(f"""
        사용자가 원하는 기간에 맞춰
        나트륨이 1000mg 이하인 식단을 추천해 주세요.
        무조건 {pdf_file_path} 또는 {docx_file_path} 자료에서만 찾아줘.
        """)
        # 저나트륨 예시 식단
        breakfast = "오트밀, 무염 견과류"
        lunch = "현미밥, 저염 닭가슴살"
        dinner = "채소 샐러드, 그릭 요거트"

    # "식단"이 포함되었지만 "고혈압"은 포함되지 않은 경우 나트륨 제한 없는 식단 추천
    elif "식단" in query and "고혈압" not in query:
        prompt = Template(f"""
        사용자가 원하는 기간에 맞춰
        나트륨 제한 없이 식단을 추천해 주세요.
        무조건 {pdf_file_path} 또는 {docx_file_path} 자료에서만 찾아줘.
        """)
        # 일반 식단 예시
        breakfast = "토스트, 계란"
        lunch = "밥, 돼지고기 불고기"
        dinner = "파스타, 샐러드"

    # "레시피"가 포함된 경우 레시피 설명
    elif "레시피" in query:
        prompt = Template(f"""
        무조건 {pdf_file_path} 또는 {docx_file_path} 자료에서 레시피를 찾아서 자세히 설명해 줘.
        """)

    # 그 외의 경우 자유로운 응답 가능
    else:
        prompt = Template(f"""
        {pdf_file_path} 또는 {docx_file_path} 자료뿐만 아니라 추가적인 정보를 포함해서 자유롭게 답변해 줘.
        """)

    return prompt.substitute(query=query)

    ''')

st.subheader("7. 대화체인 호출")

st.code('''
chain = conversation_chain
result = chain({"question": formatted_query, "chat_history": []})
response = result['answer']
source_documents = result['source_documents']
''')




