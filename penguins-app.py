import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.write(
	"""
	# ChangeGear Incident Classifier
	Write or paste your ticket description for the machine learning algorithm to predict what IT Department to assign your incident to. 
	
	"""

	)

#receive input text from user
st.subheader('Enter Incident Summary')
input_field = st.text_input("")
ticket_text = [input_field]
