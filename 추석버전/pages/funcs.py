import os 
import pandas as pd
import streamlit as st
import base64
import numpy as np
# css
def load_css(file_name):
    with open(file_name ,encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# 폰트
def font_to_base64(font_path):
    with open(font_path, "rb") as font_file:
        encoded = base64.b64encode(font_file.read()).decode('utf-8')
    return encoded

# base64로 인코딩된 폰트를 HTML로 삽입
def load_local_font(font_name, font_path):
    font_data = font_to_base64(font_path)
    font_css = f"""
    <style>
    @font-face {{
        font-family: '{font_name}';
        src: url(data:font/ttf;base64,{font_data}) format('truetype');
    }}
    html, body, [class*="css"]  {{
        font-family: '{font_name}', sans-serif;
    }}
    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)
# 줄 가로,세로
def linegaro():
    st.markdown(
        """
        <div style="border-top: 3px solid #D4BDAC; width: 100%;"></div>
        """,
        unsafe_allow_html=True)
def linesero():
    st.markdown(
        """
        <div style="border-right: 3px solid #D4BDAC; height: flex;"></div>
        """,
        unsafe_allow_html=True
    )

    