import time
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RAG import generation


def render_chunks(chunks: list):
    with st.expander(f"Pobrane fragmenty ({len(chunks)})"):
        for i, doc in enumerate(chunks, 1):
            source = doc.metadata.get("source", "nieznane")
            st.markdown(f"**Fragment {i}** — `{source}`")
            st.text(doc.page_content)
            if i < len(chunks):
                st.divider()


def render_chat(model: str, k: int):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "elapsed" in message:
                st.caption(f"Czas odpowiedzi: {message['elapsed']:.1f}s")
            if "chunks" in message:
                render_chunks(message["chunks"])

    if prompt := st.chat_input("Zadaj pytanie..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Szukam odpowiedzi..."):
                t0 = time.perf_counter()
                response, chunks = generation(prompt, model=model, k=k)
                elapsed = time.perf_counter() - t0
            st.markdown(response)
            st.caption(f"Czas odpowiedzi: {elapsed:.1f}s")
            render_chunks(chunks)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "elapsed": elapsed,
            "chunks": chunks,
        })
