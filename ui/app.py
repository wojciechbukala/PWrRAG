import streamlit as st
from sidebar import render_sidebar
from chat import render_chat

st.set_page_config(page_title="PWrRAG", page_icon="🎓", layout="wide")
st.title("Asystent PWr")

model, k = render_sidebar()
render_chat(model=model, k=k)
