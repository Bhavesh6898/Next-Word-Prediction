import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the LSTM model
model = load_model('next_word_lstm.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_length):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_length:
    token_list = token_list[-(max_sequence_length-1):] # Ensure the sequence length matches max_sequence_length
  token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
  predicted = model.predict(token_list, verbose=0)
  predicted_word_index = np.argmax(predicted, axis=1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None

# Streamlit app
st.title("Next Word Prediction using LSTM and Early Stopping")
input_text = st.text_input("Enter a sequence of words", "I love to")
if st.button("Predict Next Word"):
  max_sequence_length = model.input_shape[1]+1  # Get the input shape of the model
  next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
  if next_word:
    st.write(f"The predicted next word is: **{next_word}**")
  else:
    st.write("Could not predict the next word.")
