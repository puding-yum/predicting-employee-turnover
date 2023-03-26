import streamlit as st
import pandas as pd
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Skripsi - Judul",
    layout="centered",
    page_icon="random",
)



with st.container():
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("unnes.png")
    
    st.markdown("<h3 style='text-align: center; width:100%'>Skripsi</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center'>Penerapan Algoritma SVM dan BGWOPSO untuk Prediksi Employee Turnover</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center'>Disusun oleh: Hamid Baehaqi</h3>", unsafe_allow_html=True)

st.write('''<style>
[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    border: 5px solid black;
    padding: 0 0 2rem 0;
}

.block-container {
    padding: 3rem 1rem 3rem 1rem;
}

[data-testid="stHorizontalBlock"]{
    margin: 2rem 0 0 0;
}

</style>''', unsafe_allow_html=True)