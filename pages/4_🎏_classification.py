import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
import time
from transform_and_scale import Age, Income

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
sex = 0 if sex == 'Male' else 1

marital_status = col1.radio("Marital Status: ", ['Single', 'Non-Single'])
marital_status = 0 if marital_status == 'Single' else 1

settlement_size = col1.radio("Settlement Size: ", ['Small City', 'Mid-size City', 'Big City'])
if settlement_size == 'Small City':
    settlement_size = 0
elif settlement_size == 'Mid-size City':
    settlement_size = 1
else:
    settlement_size = 2

education = col1.selectbox("Education: ", ['No Education', 'High School', 'University', 'Graduate'])
if education == 'No Education':
    education = 0
elif education == 'High School':
    education = 1
elif education == 'Universty':
    education = 2
else:
    education = 3


occupation = col1.selectbox("Occupation: ", ['Unemployed', 'Employee/Official', 'Self-Employed/Management'])
if occupation == 'Unemployed':
    occupation = 0
elif occupation == 'Employee/Official':
    occupation = 1
else:
    occupation = 2


# ----------- COL 2 - Numerical Features -----------------------#
age = col2.slider("Age: ", 18, 100)

age = Age(age).transform_and_scale()

income = col2.slider("Income: ", 20000, 400000)
income = Income(income).transform_and_scale()




transformation_button = col2.button("Transform and Scale")
if transformation_button:
    transform_text = "Transorming and Scaling..."
    bar = col2.progress(0, text = transform_text)
    for percent_complete in range(100):
        time.sleep(0.02)
    bar.progress(percent_complete + 1, text=transform_text)
    col2.write("Age after transformation is: ")
    col2.success(str(age))
    col2.write("Income after transformation and scaling is: ")
    col2.success(str(income))






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

cluster_dict = {
    0:[["Cluster 0"], ["""
  Features:
  - Single men with medium income
  - Generally unemployed and living in small cities
  - Age is mostly between 20 and 40
  - Education is generally high school level or below
  
  """]],
  1:[['Cluster 1'],[
    """
  Features:
  - Non-single women with medium income.
  - Education level is generally high school level or above.
  - Living in small cities.
  - Either unemployed or work as employees/officials.
  """]],
  2:[['Cluster 2'],[
    """
  Features:
  - Non-single women with high income.
  - Either employeed or self-employeed.
  - Education level is generally high school level or above.
  - Living in medium to large cities.
  """
  ]],
  3:[['Cluster 3'],[
    """
  This cluster is similar to Cluster 0.
  Features:
  - Single men with higher income.
  - Management or self-employed as occupation.
  - Living in medium to large cities.
  - Education is generally high school level or below.
  """
  ]],
  4:[['Cluster 4'],[
    """
  Features:
  - Non-single males with low to medium income.
  - Education level is high school level or above.
  - Mostly employed as employees or officials.
  - Relatively equal distribution of size of city and age.

  """
  ]],
  5:[['Cluster 5'],[
    """
  Features:
  - Single females with high school level education.
  - Mostly living in small cities.
  - Either unemployed or work as employees.
  - Relatively low or medium level income.
  """
  ]]
}


if predict:
    prediction = pickled_model.predict(query)[0]
    with st.spinner("Predicting..."):
        st.header(":green[Done!]")
        st.balloons()
        time.sleep(1.5)
        result_header, result_feature = list(cluster_dict.get(prediction))
        st.header("The model predict your query to be: :red[" + result_header[0] + "]")
        with st.expander("See Features"):
            st.write(result_feature[0])




###################################################
st.sidebar.markdown("# Classification")

#################################################### hiding useless parts

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 