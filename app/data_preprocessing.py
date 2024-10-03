
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+|https\S+|email\S+', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove single characters
    text = re.sub(r'\b\w\b', '', text)
    return text.strip()

def preprocess_text(text):
    text = clean_text(text)
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column):
    df[text_column] = df[text_column].apply(preprocess_text)
    return df
