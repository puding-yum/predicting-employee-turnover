from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Skripsi - Confusion Matrix",
    layout="centered",
    page_icon="random",
)

st.markdown("<h1 style='text-align: center'>Confusion Matrix</h1>", unsafe_allow_html=True)

if 'file_name' in st.session_state:
    file_name = st.session_state['file_name']

    if file_name == "employee.csv":
        image = Image.open('./images/confusion_matrix_employee.jpg')
        # df_cf = pd.read_csv('./data/confusion_matrix.csv', index_col=0)
        # st.dataframe(df_cf)
    else:
        image = Image.open('./images/confusion_matrix_live.jpg')

    if "y_test" in st.session_state and "y_pred" in st.session_state:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accuaracy = round(accuracy_score(y_test,y_pred)*100, 2)
        precision = round(precision_score(y_test, y_pred)*100, 2)
        sensifity = round(recall_score(y_test, y_pred)*100, 2)
        f1 = round(f1_score(y_test, y_pred)*100, 2)
        specificity = round(tn/(tn+fp)*100, 2)

        col1, col2 = st.columns((2,1))
        with col1:
            st.image(image, caption=None)
        with col2:
            st.write("Accuracy = %.2f" % accuaracy)
            st.write("Precision = %.2f" % precision)
            st.write("Recall / Sensifity = %.2f" % sensifity)
            st.write("F1 = %.2f" % f1)
            st.write("Specificity = %.2f" % specificity)
    else:
        st.warning('Evaluasi data first', icon="⚠️")
else:
    st.warning('Upload data first', icon="⚠️")

st.write('''<style>
[data-testid="column"]:nth-child(2){
    padding: 2.5rem 0 0 0;
}
</style>''', unsafe_allow_html=True)