import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Clustering",
    page_icon="📈",
)

st.write("Based on the analysis conducted [here](https://github.com/furbuz/Mall-Customer-Segmentation/blob/main/Customer_Segmentation_and_Classification.ipynb) the data can be clustered into 6 Segments.")

model_code = '''
cluster_range = [i for i in range(2, 20)]
inertias = []

for c in cluster_range:
  kmeans = KMeans(n_clusters = c, random_state = 42).fit(X)
  inertias.append(kmeans.inertia_)
'''

st.code(model_code, language='python')

kmeans_image = Image.open('k-means.png')
st.image(kmeans_image)

st.subheader("Elbow method yields 6 as number of clusters")





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