import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove punctuation
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords
    text = [word for word in text if word not in stopwords.words('english')]
    text = [word for word in text if word not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

nltk.download('stopwords')

# Streamlit app code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter your message")
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    st.write(f"Transformed SMS: {transformed_sms}")  # Debugging line
    
    vector_input = tfidf.transform([transformed_sms])
    st.write(f"Vectorized Input: {vector_input}")  # Debugging line
    
    result = model.predict(vector_input)
    st.write(f"Prediction Result: {result}")  # Debugging line
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")