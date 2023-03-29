import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Mall Customer Segmentation and Clustering",
    page_icon="📈",
)



st.header("Mall Customer Data")
data = pd.read_csv("segmentation_data.csv")


st.dataframe(data)

st.write("[Data Source](https://www.kaggle.com/code/micheldc55/mall-customer-segmentation-model-interpretation/input?select=segmentation+data.csv)")
st.write("Data retrieved from [Kaggle](https://kaggle.com)")


st.header("Features")


col1, col2= st.columns(2)

col1.metric(label = "Entries", value = 2000)
col2.metric(label = "Features", value = 7)


st.header("Visualizations")

gender_data = data['Sex']
updated_data = {
    'Male' : len(gender_data[gender_data==0]),
    'Female': len(gender_data[gender_data==1])
}




###################################################
st.sidebar.markdown("# Side Bar🎈")

st.sidebar.radio("Enter age", ['Age', 'Mage'])

#################################################### hiding useless parts

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 