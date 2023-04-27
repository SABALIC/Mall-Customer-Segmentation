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


col1, col2, col3 = st.columns(3)

col1.metric(label = "Entries", value = 2000)
col2.metric(label = "Features", value = 7)
col3.metric(label = "(Future) Clusters", value = 6)

st.header("Information about Variables")

#st.subheader("Information About Variables")
data_labels = {
    "Variable" : ["ID", "Sex", "Marital status", "Age", "Education", "Income", "Occupation", "Settlement size"],
    "Data Type" : ["numerical", "categorical", "categorical", "numerical", "categorical", "numerical", "categorical", "categorical"] ,
    "Range" : ["Integer", "{0, 1}", "{0, 1}", "Integer", "{0, 1, 2, 3}", "Real", "{0, 1, 2}","{0, 1, 2}"] ,
    "Description" : ["Shows a unique identificator of a customer.",
                    "Biological sex (gender) of a customer. 0 = male / 1 = female",
                    "Marital status of a customer. 0 = single / 1 = non-single",
                    "The age of the customer in years, calculated as current year minus the year of birth of the customer at the time of creation of the dataset (Min. age = 18 / Max. age = 78)",
                    "Level of education of the customer. 0=no education / 1=high-school / 2=university / 3=graduate",
                    "Self-reported annual income in US dollars of the customer.",
                    "Category of occupation of the customer. 0=unemployed / 1=employee/oficial / 2=management or self-employed",
                    "The size of the city that the customer lives in. 0=small / 1=mid-size / 2=big"]
}

data_labels = pd.DataFrame.from_dict(data_labels)
st.dataframe(data_labels)


data.drop(columns = 'ID', inplace = True)

## ------------- Describing and getting information --- ##
#col1, col2 = st.columns(2)

st.subheader("Data Description - Numerical")

numerical_data = data[['Age', 'Income']]

st.dataframe(numerical_data.describe().T)

st.subheader("Data Description - Categorical")

categorical_data = data.drop(columns = numerical_data)

st.dataframe(categorical_data.describe().T)








###################################################
st.sidebar.markdown("# Data")

#################################################### hiding useless parts

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 