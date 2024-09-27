import streamlit as st

v
ideo_file = open('./bandicam 2024-09-27 08-50-24-796.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

st.markdown("")
