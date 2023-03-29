import streamlit as st

st.set_page_config(
    page_title="Clustering",
    page_icon="📈",
)

st.write("Based on the analysis conducted [here](https://github.com/furbuz/Mall-Customer-Segmentation/blob/main/Customer_Segmentation_and_Classification.ipynb) the data can be clustered into 6 Segments.")

col1, col2, col3 = st.columns(3)

col4, col5, col6 = st.columns(3)

#--- Segment 1 -----#
col1.header("Cluster 1")

#--- Segment 1 -----#
col2.header("Cluster 2")

#--- Segment 1 -----#
col3.header("Cluster 3")

#--- Segment 1 -----#
col4.header("Cluster 4")

#--- Segment 1 -----#
col5.header("Cluster 5")

#--- Segment 1 -----#
col6.header("Cluster 6")