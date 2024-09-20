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

with st.sidebar:
    st.subheader("메뉴")
    st.markdown("* * *")
st.markdown("* * *")
