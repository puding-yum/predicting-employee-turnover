import streamlit as st
import pandas as pd

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Skripsi - Split Data",
    layout="centered",
    page_icon="random",
)

st.markdown("<h1 style='text-align: center'>Split Data</h1>", unsafe_allow_html=True)

if 'data_train' not in st.session_state or 'data_test' not in st.session_state:
    st.warning('Preprocessing first', icon="⚠️")
else:
    data_train = st.session_state['data_train']
    data_test = st.session_state['data_test']

    tab1, tab2= st.tabs(["Data Train", "Data Testing"])
    with tab1:
        st.dataframe(data_train)
    with tab2:
        st.dataframe(data_test)