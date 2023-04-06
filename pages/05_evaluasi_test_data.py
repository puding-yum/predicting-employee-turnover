from sklearn.svm import SVC
import streamlit as st
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import pickle

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Skripsi - Evaluasi Test Data",
    layout="centered",
    page_icon="random",
)

st.markdown("<h1 style='text-align: center'>Evaluasi Test Data</h1>", unsafe_allow_html=True)

if 'file_name' in st.session_state:
    file_name = st.session_state['file_name']
    def false_predicted(data):
        return ['color: red']*len(data) if data['Predicted'] != data['Actual'] else ['color: black']*len(data)

    if 'evaluasi_test_data' in st.session_state :
        evaluasi_test_data = st.session_state['evaluasi_test_data']
        st.dataframe(evaluasi_test_data.style.apply(false_predicted, axis=1))
    elif file_name == "employee.csv":
        evaluasi_test_data = pd.read_csv("./data/evaluasi_test_data.csv")
        evaluasi_test_data = evaluasi_test_data.rename(columns={evaluasi_test_data.columns[-2]:"Actual"})
        y_test = evaluasi_test_data["Actual"]
        y_pred = evaluasi_test_data["Predicted"]
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state['evaluasi_test_data']=evaluasi_test_data 
        st.dataframe(evaluasi_test_data.style.apply(false_predicted, axis=1))
    elif "X_train" in st.session_state and "X_test" in st.session_state and "y_train" in st.session_state and "y_test" in st.session_state and "selected_features" in st.session_state and "data_test" in st.session_state :
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        selected_features = st.session_state['selected_features']
        data_test = st.session_state['data_test']
        
        svm_model = SVC(C=10, gamma=0.1, kernel='rbf')
        svm_model.fit(X_train[:, selected_features], y_train)
        y_pred = svm_model.predict(X_test[:, selected_features])
        pickle.dump(svm_model, open('./model/z-svm_model.pkl', 'wb'))

        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        
        evaluasi_test_data = data_test.copy()
        evaluasi_test_data = evaluasi_test_data.rename(columns={evaluasi_test_data.columns[-1]:"Actual"})
        evaluasi_test_data["Predicted"] = y_pred
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.savefig("./images/confusion_matrix_live.jpg")
        st.session_state['evaluasi_test_data']=evaluasi_test_data 
        st.dataframe(evaluasi_test_data.style.apply(false_predicted, axis=1))
    else:
        st.warning('Split data first', icon="⚠️")


    
    
else:
    st.warning('Upload dataset first', icon="⚠️")




