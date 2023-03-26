import streamlit as st 
import pandas as pd
from sklearn import BaseEstimator
import pickle

st.set_page_config(
    page_title="Classification",
    page_icon="📈",
)

pickled_model = pickle.load(open('model.pkl', 'rb'))










st.header("Classification of the data")

st.header("Mall Customer Segmentation and Classification Project")



st.write("Based on the data above, there are 6 clusters in the dataset")

st.write("Choose Customer features to make classification based on clusters")


# --------------------- INPUTS ----------------------------------------------#

sex = st.radio("Sex: ", ['Male', 'Female'])

marital_status = st.radio("Marital Status: ", ['Single', 'Non-Single'])

age = st.slider("Age: ", 18, 100)

education = st.selectbox("Education: ", ['No Education', 'High School', 'University', 'Graduate'])

income = st.slider("Income: ", 20000, 400000)

occupation = st.selectbox("Occupation: ", ['Unemployed', 'Employee/Official', 'Self-Employed/Management'])

settlement_size = st.radio("Settlement Size: ", ['Small City', 'Mid-size City', 'Big City'])


# ---------------------- Query ---------------------------------------------------#

query = {
    'Sex' : sex,
    'Marital status' : marital_status,	
    'Education' : education,	
    'Occupation': occupation,	
    'Settlement size' : settlement_size
}

query = pd.DataFrame.from_dict(query)
query


predict = st.button("Classify")