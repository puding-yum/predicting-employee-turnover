import streamlit as st
import numpy as np
from pickle import load
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import streamlit as st
from PIL import Image
import pandas as pd

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Skripsi - Prediksi",
    layout="centered",
    page_icon="random",
)

st.markdown("<h1 style='text-align: center'>Prediksi</h1>", unsafe_allow_html=True)

if 'file_name' in st.session_state:
    file_name = st.session_state['file_name']

    if file_name == "employee.csv":
        data_selected = st.session_state['data_selected']
        dataset = st.session_state['dataset']
        dataset = dataset[data_selected.columns]

        objList = ["Education", 'EnvironmentSatisfaction', "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"]
        for column in dataset.columns:
            if column in objList:
                dataset[column] = dataset[column].astype("object")
        data_type = dataset.dtypes

        def prediction(row_data):
            model = load(open('./model/svm_model.pkl', 'rb'))
            for idx, column_type in enumerate(data_type):
                if(idx == len(dataset.columns)-1):
                    continue

                if(column_type == "object"):
                    encoder = load(open('./model/encode/{}.pkl'.format(dataset.columns[idx]), 'rb'))
                    row_data[idx] = encoder.transform([row_data[idx]]) 
                    
                scaler = load(open('./model/scale/{}.pkl'.format(dataset.columns[idx]), 'rb'))
                scaled = scaler.transform(np.array(row_data[idx]).reshape(-1,1))
                row_data[idx] = scaled[0][0]
            
            predicted = model.predict([row_data])
            if(predicted == 1):
                st.write("Karyawan ini berpotensi meninggalkan perusahaan")
            else:
                st.write("Karyawan ini tidak berpotensi meninggalkan perusahaan")

        data_to_predict = []
        for idx, column_type in enumerate(data_type):
            if(idx == len(dataset.columns)-1):
                continue

            if(column_type == "object"):
                options = sorted(dataset.iloc[:, idx].unique().tolist())
                data_to_predict.append(st.radio("Pilih data {}!".format(dataset.columns[idx]), options=options, key=dataset.columns[idx]))
            else:
                min = dataset.iloc[:, idx].min()
                max = dataset.iloc[:, idx].max()
                data_to_predict.append(st.number_input("Isi data {}! (min = {}, max = {})".format(dataset.columns[idx], min, max), key=dataset.columns[idx], min_value=min, max_value=max))

        if st.button("Predict"):
            prediction(data_to_predict)
    else:
        st.warning('Dataset not employee', icon="⚠️")
else:
    st.warning('Upload data first', icon="⚠️")

# st.write('''<style>
# [data-testid="column"]:nth-child(2){
#     padding: 2.5rem 0 0 0;
# }
# </style>''', unsafe_allow_html=True)