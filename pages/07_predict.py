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

    # check file name
    if file_name == "employee.csv":
        form_title = [
            # "Berapa usia karyawan?", #Age
            "Seberapa sering karyawan melakukan trip bisnis?", #BusinessTravel
            "Berapa gaji harian karyawan?", #DailyRate
            "Apa departemen karyawan?", #Department
            "Berapa jarak dari tempat tinggal karyawan ke kantor?", #DistanceFromHome
            # "Apa tingkat pendidikan karyawan?", #Education
            "Apa bidang pendidikan karyawan?", #EducationField
            # "Berapa jumlah karyawan?", #EmployeeCount
            "Berapa nomor ID karyawan?", #EmployeeNumber
            "Seberapa puas karyawan terhadap lingkungan kerja?", #EnvironmentSatisfaction
            "Apa jenis kelamin karyawan", #Gender
            # "Berapa gaji perjam karyawan?", #HourlyRate
            # "Seberapa besar karyawan terlibat dalam pekerjaan?", #JobInvolvement
            "Apa tingkat pekerjaan karyawan?", #JobLevel
            "Apa posisi karyawan?", #JobRole
            "Seberapa puas karyawan terhadap pekerjaan?", #JobSatisfaction
            "Apa status pernikahan karyawan?", #MaritalStatus
            "Berapa gaji bulanan karyawan", #MonthlyIncome
            "Berapa pengeluaran perusahaan perbulan untuk karyawan?", #MonthlyRate
            "Berapa banyak perusahaan yang pernah menjadi tempat kerja karyawan?", #NumCompaniesWorked
            # "Apakah usia karyawan melebihi 18?", #Over18
            "Apakah karyawan mengambil jam lembur", #OverTime
            "Berapa kenaikan gaji karyawan dalam persen?", #PercentSalaryHike
            # "Seperti apa peringkat performa karyawan?", #PerformanceRating
            "Seberapa puas hubungan karyawan dengan karyawan lain?", #RelationshipSatisfaction
            # "Berapa standar jam kerja karyawan?", #StandardHours
            "Apa pilihan tingkat stok saham karyawan?", #StockOptionLevel
            "Berapa lama karyawan pernah bekerja?", #TotalWorkingYears
            "Berapa jumlah pelatihan yang diambil karyawan tahun lalu?", #TrainingTimesLastYear
            # "Seberapa seimbang hidup karyawan?", #WorkLifeBalance
            # "Berapa lama karyawan bekerja di perusahaan ini?", #YearsAtCompany
            "Berapa lama karyawan bekerja di posisi saat ini?", #YearsInCurrentRole
            "Berapa lama selisih tahun terakhir karyawan mendapatkan promosi dengan tahun ini?", #YearsSinceLastPromotion
            "Berapa lama karyawan bekerja dengan manajer saat ini?", #YearsWithCurrManager
            "Apakah karyawan meninggalkan perusahaan?", #Attrition
        ] 

        data_selected = st.session_state['data_selected']
        dataset = st.session_state['dataset']
        dataset = dataset[data_selected.columns]

        # change data type   
        objList = ["Education", 'EnvironmentSatisfaction', "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"]
        for column in dataset.columns:
            if column in objList:
                dataset[column] = dataset[column].astype("object")
        data_type = dataset.dtypes

        # prediction
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
                st.success("**Karyawan ini berpotensi meninggalkan perusahaan**", icon="⚠️")

            else:
                st.success("**Karyawan ini tidak berpotensi meninggalkan perusahaan**", icon="✅")

        # create form
        data_to_predict = []
        default_value = dataset.iloc[0, :]
        for idx, column_type in enumerate(data_type):
            if(idx == len(dataset.columns)-1):
                continue

            if(column_type == "object"):
                options = sorted(dataset.iloc[:, idx].unique().tolist())
                default_option = options.index(default_value[idx])
                data_to_predict.append(st.radio(
                    "{}".format(form_title[idx]), 
                    options=options, 
                    index=default_option, 
                    key=dataset.columns[idx]))
            else:
                min = dataset.iloc[:, idx].min()
                max = dataset.iloc[:, idx].max()
                data_to_predict.append(st.number_input(
                    "{} (min = {}, max = {})".format(form_title[idx], min, max), 
                    key=dataset.columns[idx], 
                    min_value=min, 
                    max_value=max,
                    value=default_value[idx]))

        if st.button("Predict"):
            prediction(data_to_predict)
    else:
        data_selected = st.session_state['data_selected']
        dataset = st.session_state['dataset']
        dataset = dataset[data_selected.columns]
        data_type = dataset.dtypes

        def prediction(row_data):
            model = load(open('./model/z-svm_model.pkl', 'rb'))
            for idx, column_type in enumerate(data_type):
                if(idx == len(dataset.columns)-1):
                    continue

                if(column_type == "object"):
                    encoder = load(open('./model/encode/z-{}.pkl'.format(dataset.columns[idx]), 'rb'))
                    row_data[idx] = encoder.transform([row_data[idx]]) 
                    
                scaler = load(open('./model/scale/z-{}.pkl'.format(dataset.columns[idx]), 'rb'))
                scaled = scaler.transform(np.array(row_data[idx]).reshape(-1,1))
                row_data[idx] = scaled[0][0]
            
            predicted = model.predict([row_data])
            if(predicted == 1):
                st.success("**Data diklasifikasikan ke dalam kelas positif**")
            else:
                st.success("**Data diklasifikasikan ke dalam  kelas negatif**")

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
    st.warning('Upload data first', icon="⚠️")
