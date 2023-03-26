# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# from scripts.skripsi import trained
# np.random.seed(42)
# st.set_page_config(
#  initial_sidebar_state="collapsed",
#  page_title="Prediksi",
#  layout="centered",
#  page_icon="random",
# )
# name_dict = {
#  'Age': 'Berapa usia anda? (tahun)',
#  'Gender': 'Jenis kelamin anda?',
#  'Polyuria': 'Apakah belakangan ini anda lebih sering buang air kecil dibanding hari biasanya?',
#  'Polydipsia': 'Apakah sering tetap merasa haus meskipun sudah minum dalam jumlah yang cukup?',
#  'sudden weight loss': 'Apakah anda mengalami penurunan berat badan secara tiba-tiba?',
#  'weakness': 'Apakah anda sering merasa lemas?',
#  'Polyphagia': 'Apakah anda sering tetap merasa lapar meskipun sudah makan dalam porsi yang cukup?',
#  'Genital thrush': 'Apakah anda mengalami infeksi jamur di alat kelamin?',
#  'visual blurring': 'Apakah anda sering mengalami pandangan mata kabur secara tiba-tiba?',
#  'Itching': 'Apakah anda sering merasa gatal/geli tanpa sebab?',
#  'Irritability': 'Apakah anda sering merasa mudah tersinggung/marah?',
#  'delayed healing': 'Apakah ketika anda memiliki luka fisik, luka tersebut tak kunjung sembuh/kering?',
#  'partial paresis': 'Apakah anda sering mengalami rasa lemah/lemas pada otot/sebagian anggota badan?',
#  'muscle stiffness': 'Apakah anda sering mengalami rasa kaku pada otot?',
#  'Alopecia': 'Apakah anda mengalami kerontokan rambut?',
#  'Obesity': 'Apakah anda mengalami obesitas/berat badan berlebih? (BMI>30)',
# }

# if "knn_smote_tmgwo_trained" not in st.session_state or not trained['akurasi']:
#  st.session_state.knn_smote_tmgwo_trained = False

# if trained['akurasi'] and st.session_state.knn_smote_tmgwo_trained:
#     data = pd.read_csv(f"datasets/{st.session_state.dataset}")

#     if data.columns[0] != "Age":
#         st.error("Ndak tahu krn bkn data diabetes", icon="🗿")
#         feature_labels = features.copy()
#         no = 1
#     else:
#         int_features = data.select_dtypes(include="int64").columns.to_list()
#         object_features = data.select_dtypes(include="object").columns.to_list()
#         selected_features = st.session_state.selected_features
#         feature_names = list(data[data.columns[np.where(selected_features)]])
#         features = list(data[data.columns[np.where(selected_features)]])
#         st.markdown("""
#         <style>
#         .css-ocqkz7{
#         margin-left: 20px;
#         </style>
#         """, unsafe_allow_html=True)
#         st.markdown("<h1 style='text-align: center; margin-bottom:20px;'>Deteksi Dini Resiko Diabetes</h1>",unsafe_allow_html=True,)
#         feature_labels = features.copy()
#         no = 1
#         for i in range(len(features)):
# if features[i] in int_features:
#     col1, col2, col3 = st.columns([1,5,1])
#     with col2:
#         features[i] = st.number_input(f"{no}. {name_dict[feature_labels[i]]}", min_value=0, max_value=100, value=30, step=1, key=feature_labels[i])
# elif features[i] in object_features:
#     col1, col2, col3 = st.columns([1,5,1])
#     with col2:
#         features[i] = st.radio(f"{no}. {name_dict[feature_labels[i]]}", 
#         options=reversed(sorted(data[features[i]].unique())), key=features[i], index=1, 
#         horizontal=True, help="Pilih salah satu")
#         no+=1
#  # else:
#  # features[i] = st.radio(f"{features[i]}", 
# options=data[features[i]].unique(), key=features[i], index=1, horizontal=True)
#  prediction = None
#  col4, col5, col6, col7, col8 = st.columns(5)
#  with col6:
#  if st.button("Prediksi", key="prediksi"):
#  model = joblib.load("models/KNN+SMOTE+TMGWO.pkl")
#  scaler = joblib.load("models/scaler.pkl")
#  X = pd.DataFrame([features], columns=feature_names)
#  X = X.replace({"Yes": 1, "No": 0,"Female": 0, "Male": 1})
#  try:
#  X["Age"] = scaler.transform([X["Age"]])
#  except KeyError:
# pass
#  prediction = model.predict(X)
#  col9,col10,col11=st.columns([1,3,1])
#  with col10:
#  if prediction is not None:
#     col10.text(f"Prediksi: {'Beresiko Terkena Diabetes' if prediction[0]==1 else 'Tidak Beresiko Terkena Diabetes'}")
#     else:
#         pass
# else:
#     st.markdown( "<h1 style='text-align: center; margin-bottom: 80px'>Prediksi</h1>",unsafe_allow_html=True, )
# st.warning("Silakan latih model KNN-SMOTE-TMGWO terlebih dahulu.", icon="⚠️")


import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pickle import load

model = load(open('./model/svm_model.pkl', 'rb'))
encoder = load(open('./model/label_encoder.pkl', 'rb'))
scaler = load(open('./model/minmaxscaler.pkl', 'rb'))

def prediction():
    # st.write(data_to_predict)
    business_travel = data_to_predict[0]
    # st.write(business_travel)
    st.write(encoder.classes_)
    # business_travel_encoded = encoder.transform([60])
    # st.write(business_travel_encoded)


data_selected = st.session_state['data_selected']
dataset = st.session_state['dataset']
dataset = dataset[data_selected.columns]

objList = ["Education", 'EnvironmentSatisfaction', "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"]
for column in dataset.columns:
    if column in objList:
        dataset[column] = dataset[column].astype("object")
# objectList = data.select_dtypes(include = "object").columns.to_list()
# nonObjectList = data.select_dtypes(exclude="object").columns.to_list()
# for x in ["Education", 'EnvironmentSatisfaction', "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"]:
#     nonObjectList.remove(x)
#     objectList.append(x)
data_type = dataset.dtypes
data_to_predict = []
array = ["dfd", "dfd", "sddd"]
# st.write(dataset.iloc[:, 0].unique().tolist())
# st.write(array)
# print(dataset.size)
# print(dataset)
for idx, column_type in enumerate(data_type):
    if(idx == len(dataset.columns)-1):
        continue

    if(column_type == "object"):
        options = sorted(dataset.iloc[:, idx].unique().tolist())
        data_to_predict.append(st.radio("Pilih data {}!".format(dataset.columns[idx]), options=options, key=dataset.columns[idx]))
        # st.write("object")
        # data_to_predict[idx] = st.radio("Select one", dataset[idx].unique())
    else:
        min = dataset.iloc[:, idx].min()
        max = dataset.iloc[:, idx].max()
        data_to_predict.append(st.number_input("Isi data {}! (min = {}, max = {})".format(dataset.columns[idx], min, max), key=dataset.columns[idx], min_value=min, max_value=max))
        # st.write("int")


if st.button("Predict"):
    prediction()

# # from model_methods import predict
# def predict(arr):
#     # Load the model
#     with open('final_model.sav', 'rb') as f:
#         model = pickle.load(f)
#     classes = {0:'Iris Setosa',1:'Iris Versicolor',2:'Iris Virginica'}
#     # return prediction as well as class probabilities
#     preds = model.predict_proba([arr])[0]
#     return (classes[np.argmax(preds)], preds)


# classes = {0:'setosa',1:'versicolor',2:'virginica'}
# class_labels = list(classes.values())
# st.title("Classification of Iris Species")
# st.markdown('**Objective** : Given details about the flower we try to predict the species.')
# st.markdown('The model can predict if it belongs to the following three Categories : **setosa, versicolor, virginica** ')
# def predict_class():
#     data = list(map(float,[sepal_length,sepal_width,petal_length, petal_width]))
#     result, probs = predict(data)
#     st.write("The predicted class is ",result)
#     probs = [np.round(x,6) for x in probs]
#     ax = sns.barplot(probs ,class_labels, palette="winter", orient='h')
#     ax.set_yticklabels(class_labels,rotation=0)
#     plt.title("Probabilities of the Data belonging to each class")
#     for index, value in enumerate(probs):
#         plt.text(value, index,str(value))
#     st.pyplot()
# st.markdown("**Please enter the details of the flower in the form of 4 floating point values separated by commas**")
# sepal_length = st.text_input('Enter sepal_length', '')
# sepal_width = st.text_input('Enter sepal_width', '')
# petal_length = st.text_input('Enter petal_length', '')
# petal_width = st.text_input('Enter petal_width', '')
# if st.button("Predict"):
#     predict_class()