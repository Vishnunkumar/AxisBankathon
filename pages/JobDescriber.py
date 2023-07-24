import pandas as pd
import re
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import random
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

r_pattern = "\w+"
st.title("CV Recommender")
st.markdown('Way to optimize HR Analytics')

form = st.form("Resume submit form")
input_text = form.text_input(label='Enter Job descriptions')
submit = form.form_submit_button("Submit")

r_pattern = "\w+"
(' ').join(re.findall(r_pattern, input_text))

# Load model
embed = hub.load("https://tfhub.dev/google/nnlm-en-dim50/2")

def get_embeds(text):
    return embed([text]).numpy()

# Load titles
df = pd.read_csv('archive/data job posts.csv')
titles = list(df.Title.dropna(axis=0))
title_emdeds = [get_embeds(x) for x in titles[0:3000]]
dist = DistanceMetric.get_metric('euclidean')

# Similarity Search 
def find_similar(query):
    best_ = []
    query_embed = get_embeds(query)
    i = 0
    for i in range(0, len(title_emdeds)):
        score = dist.pairwise(title_emdeds[i], query_embed)[0][0]
        best_.append([i ,score])
        i += 1
    
    best_.sort(key=lambda x: x[1])
    return best_[0:5]

jobs = find_similar(input_text)
job_titles = [titles[x[0]] for x in jobs]
df_job = pd.DataFrame(job_titles)
df_job.columns = ['Recommended Jobs']
if submit:
    st.table(df_job)
    txt = st.text_input('Provide any suggesstions')
    job_description = txt
