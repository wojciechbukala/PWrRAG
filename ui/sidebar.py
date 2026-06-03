import streamlit as st

AVAILABLE_MODELS = [
    "gemma3:4b",
    "gemma3:1b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "llama3.2:1b",
]


def render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        st.header("Ustawienia")

        model = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=0,
        )

        k = st.slider(
            "Liczba chunków (k)",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
        )

        st.divider()
        if st.button("Wyczyść historię czatu"):
            st.session_state.messages = []
            st.rerun()

    return model, k
