import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go




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




## ------------- Describing and getting information --- ##
#col1, col2 = st.columns(2)

st.dataframe(data.drop(columns ="ID").describe())




## ----------- Visualizations ---------------- ##
st.header("Visualizations")


## -------------------------------------------------- INCOME
st.subheader("Income Distribution")
income = data['Income']


marker_color = '#657220'  # Orange color

fig = go.Figure()
fig.add_trace(go.Histogram(x=income, marker_color=marker_color))

fig.update_layout(
    xaxis_title_text='Income',
    yaxis_title_text='Count'
)

st.plotly_chart(fig)

## -------------------------------------------------- AGE
st.subheader("Age Distribution")
age = data['Age']

fig = px.histogram(age, nbins=40)
fig.update_layout(xaxis_title_text = 'Age')
st.plotly_chart(fig)

## -------------------------------------------------- GENDER
st.subheader("Gender")
male = sum(data['Sex'])
female = len(data['Sex']) - male
gender_data = {
    "Male": male,
    "Female": female
}

df = pd.DataFrame(gender_data.items(), columns=['Gender', 'Count'])

figure = px.bar(df, x='Gender', y='Count', color='Gender')

st.plotly_chart(figure)

## -------------------------------------------------- MARITAL
st.subheader("Marital Status")
married = sum(data['Marital status'])
non_married = len(data['Marital status']) - married
marital_status_data = {
    "Married": married,
    "Non-Married": non_married
}

x_data = list(marital_status_data.keys())
y_data = list(marital_status_data.values())

# Create trace for the bar chart
trace = go.Bar(x=x_data, y=y_data, marker=dict(color=['blue', 'violet']))


# Create the figure object
fig = go.Figure(data=[trace])

# Render the chart using Plotly in Streamlit
st.plotly_chart(fig)

## -------------------------------------------------- OCCUPATION
st.subheader("Occupation")

# Define a list of colors for each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create a bar plot of the number of customers in each occupation category
occupation_counts = data['Occupation'].value_counts()
fig = go.Figure(go.Bar(x=['Unemployed', 'Employee/official', 'Management/self-employed'], y=occupation_counts, marker_color=colors))
fig.update_layout( xaxis_title="Occupation", yaxis_title="Number of Customers")

# Show the plot in Streamlit
st.plotly_chart(fig)


## -------------------------------------------------- SETTLEMENT
st.subheader("Settlement Size")
# Define a list of colors for each bar
colors = ['#dd1c77', '#beaed4', '#91cf60']

# Create a bar plot of the number of customers in each settlement size category
settlement_counts = data['Settlement size'].value_counts()
fig = go.Figure(go.Bar(x=['Small', 'Mid-size', 'Big'], y=settlement_counts, marker_color=colors))
fig.update_layout(xaxis_title="Settlement Size", yaxis_title="Number of Customers")

# Show the plot in Streamlit
st.plotly_chart(fig)

## -------------------------------------------------- EDUCATION
st.subheader("Education")

# Define a list of colors for each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create a bar plot of the number of customers in each education category
edu_counts = data['Education'].value_counts()
fig = go.Figure(go.Bar(x=['No Education', 'High School', 'University', 'Graduate'], y=edu_counts, marker_color=colors))
fig.update_layout(xaxis_title="Education Level", yaxis_title="Number of Customers")

# Show the plot in Streamlit
st.plotly_chart(fig)


















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