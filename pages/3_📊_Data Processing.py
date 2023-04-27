import streamlit as st
import pandas as pd
import time

st.set_page_config(
    page_title="Processing",
    page_icon="🔍",
)

data = pd.read_csv('./segmentation_data.csv')
transform_and_scale_data = pd.read_csv('./classification_data.csv')
transform_and_scale_data.drop(columns=['Labels'], inplace = True)

st.header("Raw Data")
st.dataframe(data)

st.subheader("The data needs to be cleaned, transformed and scaled for clustering process")
transform_scale_data = st.button("Clean, Transform and Scale")
if transform_scale_data:
   with st.spinner("Transforming and Scaling entire dataset..."):
        time.sleep(3)
        st.dataframe(transform_and_scale_data)

        st.write("The data has been cleaned, transformed and scaled based on the analysis conducted in the previous step.")
        st.subheader("The Age has been transformed by :blue[Log Transformation]")
        st.subheader("The Income has been transformed by :blue[Power Transformation]")
        st.subheader("After that, the entire dataset has been transformed by :red[MinMaxScaler()]")





###################################################
st.sidebar.markdown("# Processing")

#################################################### hiding useless parts

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 