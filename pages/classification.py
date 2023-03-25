import streamlit as st

st.set_page_config(
    page_title="Classification",
    page_icon="📈",
)

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





predict = st.button("Classify")