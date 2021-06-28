import streamlit as st
import pandas as pd
import numpy as np 
import nltk 
import regex
import pickle 
from PIL import Image
from nltk.corpus import stopwords
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


st.write(
	"""
	# ChangeGear Incident Classifier
	Write or paste your ticket description for the machine learning algorithm to predict what IT Department to assign your incident to. Test 14:23
	
	"""

	)

#receive input text from user
st.subheader('Enter Incident Summary')
input_field = st.text_input("")
ticket_text = [input_field]

#import in our trained model
with open('model.pickle', 'rb') as f: 
    model = pickle.load(f)

wpt=nltk.WordPunctTokenizer()
stop_words=nltk.corpus.stopwords.words('english')

#define pipeline function for text normalization
def normalize_doc(doc):
    doc=regex.sub(r'[^a-zA-Z\s]', '', doc) 
    doc=doc.lower() 
    doc=doc.strip() 
    tokens=wpt.tokenize(doc)
    filtered_tokens=[token for token in tokens if token not in stop_words]
    doc=' '.join(filtered_tokens)
    return doc

#apply normalization pipeline to input text
normalize_corpus=np.vectorize(normalize_doc) #create a vectorized object for our normalization pipeline
norm_text=normalize_corpus(ticket_text) #clean and normalize the ticket

#access parts of our model for implementation
vect=model.named_steps['tfidf']
clf=model.named_steps['clf']

trans_text=vect.transform(norm_text).toarray() #perform text extraction on text
prediction=clf.predict(trans_text) #predict class using logistic regression
probabilities=(clf.predict_proba(trans_text)) #predict probabilities using logistic regression
