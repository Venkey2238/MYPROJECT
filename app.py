import streamlit as st
import numpy as np
import pickle
import keras
import tensorflow_hub as hub
import pandas as pd
import tensorflow_text as text
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltp import Preprocessor
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix , classification_report
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


model = pickle.load(open('/home/mr1ncr1d1ble/github/SpamEmailDetection-NLP-Project/APP/FED_pickle','rb'))
def predict_Email(docx):
    test_results = model.predict([docx])
    output = np.where(test_results>0.4,'spam', 'ham')
    return output
def main():
    st.title("Fraud E-mail Detection model")
    menu = ["Home","Monitor","About"]

    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("SPAM E-mail in Text")
        with st.form(key="myform"):
            raw_text = st.text_area("Copy your E-mail subject or body Here")
            submit_text = st.form_submit_button(label="Submit")
        if submit_text:

            #Apply functions here
            prediction = predict_Email(raw_text)
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            st.write(prediction)
                   

    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")


if  __name__ == '__main__':
    main()
