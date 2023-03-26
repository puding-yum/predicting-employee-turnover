import functools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
from BGWOPSO import BGWOPSO

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Skripsi - Preprocessing",
    layout="centered",
    page_icon="random",
)

st.markdown("<h1 style='text-align: center'>Preprocessing</h1>", unsafe_allow_html=True)

if 'data_cleaned' in st.session_state and 'data_encoded' in st.session_state and 'data_normalized' in st.session_state and 'data_selected' in st.session_state:
    data_cleaned = st.session_state['data_cleaned']
    data_encoded = st.session_state['data_encoded']
    data_normalized = st.session_state['data_normalized']
    data_selected = st.session_state['data_selected']

    
elif 'dataset' in st.session_state:
    file_name = st.session_state['file_name']
    dataset = st.session_state['dataset']

    if file_name == "employee.csv":
        data_cleaned = dataset
        data_encoded = pd.read_csv('./data/data_encoded.csv')
        data_normalized = pd.read_csv('./data/data_normalized.csv')
        data_selected = pd.read_csv('./data/data_selected.csv')
        data_train = pd.read_csv("./data/data_train.csv")
        data_test = pd.read_csv("./data/data_test.csv")
        
        st.session_state['data_train'] = data_train
        st.session_state['data_test'] = data_test
        st.session_state['data_selected']=data_selected
    else:
        # data cleaning
        data_cleaned = dataset.dropna()
        st.session_state['data_cleaned'] = data_cleaned
    
        # label encoder
        objectList = data_cleaned.select_dtypes(include = "object").columns.to_list()
        nonObjectList = data_cleaned.select_dtypes(exclude="object").columns.to_list()

        le = LabelEncoder()
        data_encoded = dataset.copy()
        for object in objectList:
            data_encoded[object] = le.fit_transform(data_encoded[object])
        st.session_state['data_encoded']=data_encoded

        # normalization
        scaler = MinMaxScaler()
        data_normalized = data_encoded.copy()
        for column in data_normalized.columns:
            data_normalized[column] = scaler.fit_transform(data_normalized[column].values.reshape(-1,1))
        st.session_state['data_normalized']=data_normalized 

        X = data_normalized.drop(data_normalized.columns[-1],axis=1).values
        y = data_normalized[data_normalized.columns[-1]].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
        st.session_state['X_train']=X_train
        st.session_state['X_test']=X_test
        st.session_state['y_train']=y_train
        st.session_state['y_test']=y_test

        def fitness(x, X_train, X_test, y_train, y_test):
            alpha = 0.99
            beta = 1-alpha
            if x.ndim == 1:
                x = x.reshape(1, -1)
            loss = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if np.sum(x[i, :]) > 0:
                    model = SVC(C=10, gamma=0.1, kernel='rbf')
                    model.fit(X_train[:, x[i,:].astype(bool)], y_train)
                    acc = model.score(X_test[:, x[i,:].astype(bool)], y_test)
                    error_rate = 1 - acc
                    loss[i] = alpha * error_rate + beta * (np.sum(x[i, :]) / X.shape[1])
                else:
                    loss[i] = np.inf
            return loss


        lossfunc = functools.partial(fitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        optimizer = BGWOPSO(lossfunc, X_train.shape[1], 8, 70)
        optimizer.opt()
        selected_features = optimizer.gBest_X.astype(bool)
        data_selected = data_normalized.loc[:, np.append(selected_features, True)]
        st.session_state['data_selected']=data_selected
        st.session_state['selected_features']=selected_features
        
        data_train = pd.DataFrame(X_train[:, selected_features])
        data_train[data_normalized.columns[-1]] = y_train
        data_train.columns = data_selected.columns

        data_test = pd.DataFrame(X_test[:, selected_features])
        data_test[data_normalized.columns[-1]] = y_test
        data_test.columns = data_selected.columns

        st.session_state['data_train']=data_train
        st.session_state['data_test']=data_test

    tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning", "Label Encoder", "Normalization", "Feature Selection"])
    with tab1:
        st.dataframe(data_cleaned)
    with tab2:
        st.dataframe(data_encoded)
    with tab3:
        st.dataframe(data_normalized)
    with tab4:
        st.dataframe(data_selected)
else:
    st.warning('Upload dataset first', icon="⚠️")
    