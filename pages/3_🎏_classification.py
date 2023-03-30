import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
import transform_and_scale as ts

st.set_page_config(
    page_title="Classification",
    page_icon="📈",
)

pickled_model = joblib.load('model.pkl')










st.header("Classification of the data")

st.header("Mall Customer Segmentation and Classification Project")



st.write("Based on the data above, there are 6 clusters in the dataset")

st.write("Choose Customer features to make classification based on clusters")


# --------------------- INPUTS ----------------------------------------------#

col1, col2 = st.columns(2)

# ------------ COL 1 - Categorical Features --------------------#

sex = col1.radio("Sex: ", ['Male', 'Female'])

marital_status = col1.radio("Marital Status: ", ['Single', 'Non-Single'])

settlement_size = col1.radio("Settlement Size: ", ['Small City', 'Mid-size City', 'Big City'])

education = col1.selectbox("Education: ", ['No Education', 'High School', 'University', 'Graduate'])

occupation = col1.selectbox("Occupation: ", ['Unemployed', 'Employee/Official', 'Self-Employed/Management'])



# ----------- COL 2 - Numerical Features -----------------------#
age = col2.slider("Age: ", 18, 100)
age = ts.Age(age).transform_and_scale()

income = col2.slider("Income: ", 20000, 400000)
income = ts.Income(income).transform_and_scale()






# ---------------------- Query ---------------------------------------------------#

query = {
    'Income':[income],
    'Age':[age],
    'Sex' : [sex],
    'Marital status' : [marital_status],	
    'Education' : [education],	
    'Occupation': [occupation],	
    'Settlement size' : [settlement_size]
}

query = pd.DataFrame.from_dict(query)
print("QUERY", query)
predict = st.button("Classify")

if predict:
    prediction = pickled_model.predict(query)
    print("PREDICTION:", prediction)