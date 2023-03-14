import time
import streamlit as st
import numpy as np
import pandas as pd
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
import requests
import os
import string
import tensorflow
from tensorflow import keras

import pickle
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers


st.set_page_config(page_title="Indonesian News Title Category Classifier",
                   page_icon="🗞️", layout="centered")


@st.cache(allow_output_mutation=True, show_spinner=False, ttl=3600, max_entries=10)
def build_model():
    with st.spinner("Loading models... this may take awhile! \n Don't stop it!"):
        model = keras.models.load_model('lstm_model_modifikasi.h5')
        with open('tokenizer.pickle', 'rb') as f:
            Tokenizer = pickle.load(f)
        inference = model, Tokenizer
    return inference


inference, Tokenizer = build_model()

st.title('🗞️ Indonesian News Title Category Classifier')

with st.expander('📋 About this app', expanded=True):
    st.markdown("""
    * Indonesian News Title Category Classifier app is an easy-to-use tool that allows you to predict the category of a given news title.
    * You can predict one title at a time or upload .csv file to bulk predict.
    * Made by [Alpian Khairi](https://www.linkedin.com/in/alpiankhairi/), [Sheva Satria](), [Fernandico](), [Bagus]().
    """)
    st.markdown(' ')

with st.expander('🧠 About prediction model', expanded=False):
    st.markdown("""
    ### Indonesian News Title Category Classifier
    * Model are trained using [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) based on [Indonesian News Title Dataset]() from  on .
    * Model test accuracy is **~93%**.
    * **[Source Code]()**
    """)
    st.markdown(' ')


st.markdown(' ')
st.markdown(' ')

st.header('🔍 News Title Category Prediction')

title = st.text_input(
    'News Title', placeholder='Enter your shocking news title + its narrative')

if title:
    with st.spinner('Loading prediction...'):
        tokenized_test = Tokenizer.texts_to_sequences(title)
        X_test = pad_sequences(tokenized_test, maxlen=250)
        result = inference.predict(X_test)
        pred_labels = []
        for i in result:
            if i > 0.5:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        for i in range(len(title)):
            if pred_labels[i] == 1:
                s = 'Fact'
            else:
                s = 'Hoax'
    st.markdown(f'Category for this news is **[{s}]**')


st.markdown(' ')
st.markdown(' ')

st.header('🗃️ Bulk News Title Category Prediction')
st.markdown(
    'Only upload .csv file that contains list of news titles separated by comma.')

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    results = []
    # df = df.tail(20)

    with st.spinner('Loading prediction...'):
        for title in df['narasi']+df['judul']:
            tokenized_test = Tokenizer.texts_to_sequences(title)
            X_test = pad_sequences(tokenized_test, maxlen=250)
            result = inference.predict(X_test)
            pred_labels = []
            for i in result:
                if i > 0.5:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

            for i in range(len(title)):
                if pred_labels[i] == 1:
                    s = 'Hoax'
                else:
                    s = 'Fact'
            results.append({'Title': title, 'Category': s})

        df_results = pd.DataFrame(results)

    st.markdown('#### Prediction Result')
    st.download_button(
        "Download Result",
        df_results.to_csv(index=False).encode('utf-8'),
        "News Title Category Prediction Result.csv",
        "text/csv",
        key='download-csv'
    )
    st.dataframe(df_results, 1000)
