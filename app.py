import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import io

 #Load the word index from the IMDB dataset
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

##Helper functions

## Function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])


### Function to preprocess user input

def preprocess_text(text):
    max_vocab_size = 10000

    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 2) + 3, max_vocab_size - 1) for word in words]
    # encoded_review= [word_index.get(word,0) for word in words]
    # encoded_review = [word_index.get(word,2)+3 for word in words ]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=750)
    return padded_review


## Streamlit

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.file_uploader(label='Upload CSV file', type ='csv',accept_multiple_files=False)

if user_input is not None:
    input_df = pd.read_csv(user_input)

    if 'review' in input_df.columns:
        sentiments = []
        scores = []

    for review in input_df['review']:
        preprocess_rev = preprocess_text(review)
        prediction = model.predict(preprocess_rev)
        sentiment = 'positive' if prediction >= 0.5 else 'negative'
        sentiments.append(sentiment)


    input_df['sentiment'] = prediction


    buffer = io.BytesIO()
    input_df.to_csv(buffer, index=False)
    buffer.seek(0)

    st.download_button(
            label="Download Predictions as CSV",
            data=buffer,
            file_name="sentiment_predictions.csv",
            mime="text/csv"
        )
else:
    st.write("Error: The uploaded CSV file must contain a 'review' column.")
   





