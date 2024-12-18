import os

print(os.path.isdir("templates"))
print(os.path.isfile("templates/index.html"))

from flask import Flask, render_template, request
import pickle
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorization.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'\W', ' ', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'<.*?>+', '', text)  
    text = re.sub(f'[{string.punctuation}]', '', text)  
    text = re.sub(r'\n', '', text) 
    text = re.sub(r'\w*\d\w*', '', text)  
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']  
        cleaned_news = clean_text(news)  
        vect_news = vectorizer.transform([cleaned_news])  
        prediction = model.predict(vect_news) 
        result = "Fake News" if prediction[0] == 0 else "Real News"
        return render_template('index.html', result=result)  

if __name__ == '__main__':
    app.run(debug=True)
