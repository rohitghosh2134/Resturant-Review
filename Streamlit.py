import pandas as pd
import numpy as np
import joblib
import streamlit as st
import re,string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
def preprocessor(data):
    sw= stopwords.words('english')
    sw.remove('not')
    sw.remove("don't")
    sw.remove("shouldn't")
    sw.remove("wouldn't")
    lemma=WordNetLemmatizer()
    ps = PorterStemmer()
    
    re.sub(r"http\S+|www.*", "", data)#remove url links
    data = word_tokenize(data.lower())#lowercase and tokenize the words
    data = [word for word in set(data) if word.isalpha() and word not in string.punctuation and word!='\n' and len(word) > 2 and word not in sw]

    return " ".join(data)



cv = joblib.load(open('cv.pkl', 'rb'))

load_model=joblib.load('finalmodel.sav')

def prediction(Review):
    pred=load_model.predict([Review])
    
    if pred==1:
        pred="Postive"
    else:
        pred="Negative"
    return pred

html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Resturant Review</h1> 
    </div> 
    """
      
    # display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 

Review=st.text_input("Enter Review")

result=" "

if st.button("Predict"):
    result=prediction(Review)
    st.success("Your Review is {}".format(result))
    
