import streamlit as st
import pandas as pd
import joblib
import time
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


## -------------- PRINCIPAL COMPONENT ANALYSIS ----------------------------##
st.header("The result of Principal Component Analysis yields such graph:")

with st.expander("See Explanation"):
  st.write("""
    Principal Component Analysis (PCA) is a statistical technique used to analyze and summarize large datasets. 
    In simpler terms, PCA is a way to simplify data and reduce noise by taking a large set of variables and summarizing them into fewer, more meaningful variables called principal components. 
    These principal components represent the patterns of variability in the original dataset and can be used to visualize and analyze the data in a more efficient and effective way. 
    Essentially, PCA helps us to understand the most important aspects of a dataset and make sense of complex information.
  
  """)
  st.image("https://miro.medium.com/v2/resize:fit:596/1*QinDfRawRskupf4mU5bYSA.png")


X = pd.read_csv('classification_data.csv')


pca = PCA(n_components = 3, random_state = 666)
X_pca = pca.fit_transform(X)

X_pca_df = pd.DataFrame(data = X_pca, columns = ['X1', 'X2', 'X3'])

kmeans = KMeans(n_clusters = 6, random_state = 0).fit(X)

labels = kmeans.labels_
X_pca_df['Labels'] = labels

X_pca_df['Labels'] = X_pca_df['Labels'].astype(str)

fig = px.scatter_3d(X_pca_df, x='X1', y='X2', z='X3',
              color=X_pca_df['Labels'])

st.plotly_chart(fig)


## ------------ END OF PRINCIPAL COMPONENT ANALYSIS --------------------##





## --------------------------- heptagon Radar Chart ----------------------##


### --------------------- End of HEPTAGON RADAR CHART --------------------------#


st.write("Choose the algorithm:")
st.selectbox("", ['K Means'])

num = st.selectbox("Choose number of Clusters: ", [2, 3, 4, 5, 6, 7, 8, 9])

fun_button = st.button("Click to run the clustering process!!!")
if fun_button:
  if num != 6:
    st.write("You should have chosen 6! Luckily, 6 was already chosen.")
  with st.spinner("Clustering the data..."):
      time.sleep(3)

      st.header(":green[DONE SUCCESSFULLY!!!]")


      st.balloons()

      col1, col2, col3 = st.columns(3)

      col4, col5, col6 = st.columns(3)

      #--- Segment 0 -----#
      col1.header("Cluster 0")
      col1.write(
        """
        Features:
        - Single men with medium income
        - Generally unemployed and living in small cities
        - Age is mostly between 20 and 40
        - Education is generally high school level or below
        
        """
      )

      #--- Segment 1 -----#
      col2.header("Cluster 1")
      col2.write(
        """
        Features:
        - Non-single women with medium income.
        - Education level is generally high school level or above.
        - Living in small cities.
        - Either unemployed or work as employees/officials.
        """
      )

      #--- Segment 2 -----#
      col3.header("Cluster 2")
      col3.write(
        """
        Features:
        - Non-single women with high income.
        - Either employeed or self-employeed.
        - Education level is generally high school level or above.
        - Living in medium to large cities.
        """
      )

      #--- Segment 3 -----#
      col4.header("Cluster 3")
      col4.write(
        """
        This cluster is similar to Cluster 0.
        Features:
        - Single men with higher income.
        - Management or self-employed as occupation.
        - Living in medium to large cities.
        - Education is generally high school level or below.
        """
      )

      #--- Segment 4 -----#
      col5.header("Cluster 4")
      col5.write(
        """
        Features:
        - Non-single males with low to medium income.
        - Education level is high school level or above.
        - Mostly employed as employees or officials.
        - Relatively equal distribution of size of city and age.

        """
      )

      #--- Segment 5 -----#
      col6.header("Cluster 5")
      col6.write(
        """
        Features:
        - Single females with high school level education.
        - Mostly living in small cities.
        - Either unemployed or work as employees.
        - Relatively low or medium level income.
        """
      )



###################################################
st.sidebar.markdown("# Clustering")

#################################################### hiding useless parts

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 