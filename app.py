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
import train as tr
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers


st.set_page_config(page_title="Indonesian News Title Category Classifier",
                   page_icon="🗞️", layout="centered")


@st.cache_resource()
def build_model():
    with st.spinner("Loading models... this may take awhile! \n Don't stop it!"):
        model = keras.models.load_model('C:/Users/lenovo/PycharmProjects/pythonProject/Sentiment-Analysis/lstm_model.h5')
        with open('C:/Users/lenovo/PycharmProjects/pythonProject/Sentiment-Analysis/tokenizer.pickle', 'rb') as f:
            Tokenizer = pickle.load(f)
        inference = model, Tokenizer
    return inference


inference, Tokenizer = build_model()

st.title('🏨 Indonesian Hotel Review Sentiment Analysis')

with st.expander('📋 Tentang App ini', expanded=False):
    st.markdown("""
    * Indonesian Hotel Review app adalah alat yang mudah digunakan yang memungkinkan Anda memprediksi kategori ulasan hotel yang diberikan.
    * Anda hanya dapat memprediksi satu ulasan dalam satu waktu.
    * Dibuat oleh Dini Andriani, Damar Fadhil Muhammad, Ichiro Gabriel Rivaldo S., M. Redho Dermawan, Muhamad Akbar.
    """)
    st.markdown(' ')

with st.expander('🧠 Tentang model', expanded=False):
    st.markdown("""
    ### Indonesian Hotel Review Sentiment Analysis
    * Model dilatih menggunakan [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) berdasarkan [Indonesian Hotel Review Dataset](https://huggingface.co/datasets/rakkaalhazimi/hotel-review)
    * Akurasi model adalah **~98%**.
    * **[Source Code](https://github.com/dfmuh/Analisis-Sentimen-Review-Hotel)**
    """)
    st.markdown(' ')


st.markdown(' ')
st.markdown(' ')

st.header('🔍 Hotel Review Prediction')

review = st.text_input(
    'Review', placeholder='Enter hotel review')

if review:
    with st.spinner('Loading prediction...'):
        # tokenized_test = Tokenizer.texts_to_sequences(title)
        # X_test = pad_sequences(tokenized_test, maxlen=200)
        # result = inference.predict(X_test)
        # pred_labels = []
        # for i in result:
        #     if i > 0.5:
        #         pred_labels.append(1)
        #     else:
        #         pred_labels.append(0)

        # for i in range(len(title)):
        #     if pred_labels[i] == 1:
        #         s = 'Fact'
        #     else:
        #         s = 'Hoax'

        # tw = Tokenizer.texts_to_sequences([text])
        # tw = pad_sequences(tw,maxlen=200)
        # prediction = int(model.predict(tw).round().item())
        # if sentiment_label[1][prediction] == 0:
        #     result = "Negatif"
        # else :
        #     result = "Positif"
        # print("Hasil Prediksi : ", result)

        s = tr.predict_sentiment(review)
        st.markdown(f'Category for this news is **[{s}]**')


st.markdown(' ')
st.markdown(' ')