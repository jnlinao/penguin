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

#Display Niagara logo
image = Image.open('niagara_logo.jpg')
st.image(image, width=250)

st.write(
	"""
	# ChangeGear Incident Classifier
	Write or paste your ticket description for the machine learning algorithm to predict what IT Department to assign your incident to. 
	
	"""

	)

#add sidebar panel
st.sidebar.header("Welcome to ChangeGear Incidient Classifier")
st.sidebar.write(
    """
    ## Purpose
   
    This prototype application serves to supplement the clarity of which IT Department teams should be responsible of your submitted ChangeGear request.
    Due to the misclassification of tickets, many teams face an overload of "out of scope" tasks, resulting in a misalignment of responsibilities between teams and spent resources. 

    ## How It Works
    This app is built on top of a machine learning algorithm. The model has been trained with thousands of records of ChangeGear data in order to make the most precise prediction.
    When a user inputs their incident issue, a prediction is provided. Below the predicted team is a probability chart of the prediction. This is done by Logistic Regression coefficients.
    Please be as descriptive in your message as possible, avoiding any typos. It is important to note that this algorithm is still early in its development.

    [See Github Repository](https://github.com/jnlinao/penguin)

    #### Developed by Jacob Linao
    #### Contact: [jacob7141299@gmail.com](mailto:john@example.com)

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

#categorize class prediction corresponding with owner then output
def output_prediction(label_prediction):
    #identify the class prediction 
    if label_prediction==0:
        owner='EAM & MRO Inventory Support'
    elif label_prediction==1:
        owner='HRIS Team'
    elif label_prediction==2: 
        owner='IT Service Desk Team'
    elif label_prediction==3:
        owner='Network Team'
    elif label_prediction==4:
        owner='OTM Support'
    elif label_prediction==5:
        owner='Planning Support'
    elif label_prediction==6:
        owner='System Admin Team'
    else:
        owner='WMS Team'
    return		(owner)

#format dataframe of predictions
def output_probabilities(array_prob):
    prob_classes=array_prob[0][:]
    prob_classes_num=[round(x,3) for x in prob_classes]
    
    classes = ['EAM & MRO Inventory Support', 'HRIS Team', 'IT Service Desk Team', 'Network Team', 'OTM Support', 
          'Planning Support', 'System Admin Team', 'WMS Team']
    ownerDict_num = dict(zip(classes, prob_classes_num))
    ownerDf_num = pd.DataFrame(ownerDict_num.items(), columns=['Owner', 'Probability'])
    
    return(ownerDf_num)


st.subheader('Model Team Prediction:')
st.write(output_prediction(prediction))

st.subheader('Model Team Probabilities:')
st.write('Shows the probabilities of your ticket belonging to each team')
prop_df = output_probabilities(probabilities)
fig=px.bar(prop_df, x='Owner', y='Probability')
st.plotly_chart(fig)

