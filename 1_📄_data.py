import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Mall Customer Segmentation and Clustering",
    page_icon="📈",
)



st.header("Mall Customer Data")
data = pd.read_csv("segmentation_data.csv")


st.dataframe(data)

st.header("Features")

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Side Bar🎈")

st.sidebar.radio("Enter age", ['Age', 'Mage'])

