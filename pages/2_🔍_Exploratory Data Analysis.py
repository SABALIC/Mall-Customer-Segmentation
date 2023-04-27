import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import scipy.stats as stats
import scipy.special
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer




st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📈",
)

# reading data
data = pd.read_csv("segmentation_data.csv")

# divider function
def divider():
    return st.write("<hr>", unsafe_allow_html=True)


# Sidebar
###################################################
st.sidebar.markdown("# Exploratory Data Analysis")



st.header("Exploratory Data Analysis")

page_names = ['Univariate Analyis', 'Bivariate Analysis', 'Multivariate Analysis', 'General Conclusion', 'Feature Transforming and Scaling']
page = st.sidebar.radio("Navigation", page_names)

st.header(":red["+page+"]")

if page == 'Univariate Analyis':
    
    if st.button("Take me to conclusion/insights!"):
        pass
    st.subheader("Distribution of the Numeric Variables")
    
    ## -------------------------------------------------- INCOME
    st.subheader("_Income Distribution_")
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
    st.subheader("_Age Distribution_")
    age = data['Age']

    fig = px.histogram(age, nbins=40)
    fig.update_layout(xaxis_title_text = 'Age')
    st.plotly_chart(fig)
    
    
    st.subheader("Distribution of the Categorical Variables")
    
    ## -------------------------------------------------- GENDER
    st.subheader("_Gender_")
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
    st.subheader("_Marital Status_")
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
    st.subheader("_Occupation_")

    # Define a list of colors for each bar
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create a bar plot of the number of customers in each occupation category
    occupation_counts = data['Occupation'].value_counts()
    fig = go.Figure(go.Bar(x=['Unemployed', 'Employee/official', 'Management/self-employed'], y=occupation_counts, marker_color=colors))
    fig.update_layout( xaxis_title="Occupation", yaxis_title="Number of Customers")

    # Show the plot in Streamlit
    st.plotly_chart(fig)


    ## -------------------------------------------------- SETTLEMENT
    st.subheader("_Settlement Size_")
    # Define a list of colors for each bar
    colors = ['#dd1c77', '#beaed4', '#91cf60']

    # Create a bar plot of the number of customers in each settlement size category
    settlement_counts = data['Settlement size'].value_counts()
    fig = go.Figure(go.Bar(x=['Small', 'Mid-size', 'Big'], y=settlement_counts, marker_color=colors))
    fig.update_layout(xaxis_title="Settlement Size", yaxis_title="Number of Customers")

    # Show the plot in Streamlit
    st.plotly_chart(fig)

    ## -------------------------------------------------- EDUCATION
    st.subheader("_Education_")

    # Define a list of colors for each bar
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create a bar plot of the number of customers in each education category
    edu_counts = data['Education'].value_counts()
    fig = go.Figure(go.Bar(x=['High School','University', 'No Education', 'Graduate'], y=edu_counts, marker_color=colors))
    fig.update_layout(xaxis_title="Education Level", yaxis_title="Number of Customers")

    # Show the plot in Streamlit
    st.plotly_chart(fig)
    
    
    st.subheader("_Results/Conclusions_")
        
    points = [
        'The gender of the customers seem to be relatively balanced.', 
        'The martial status of the customers seem balanced, as well.', 
        'Most of the customers are educated, with majority of them having at least high school level education.',
        'There are sufficient numbers of each instances in \'Occupation\' and \'Settlement Size\' features.'
    ]

    for point in points:
        st.markdown("- " + point)
    
    
elif page == 'Bivariate Analysis':
    
    if st.button("Take me to conclusion/insights!"):
        pass
    
    st.subheader("Numerical vs Numerical")
    scatter_data = data[['Age', 'Income']]
    fig = px.scatter(scatter_data, x = 'Age', y = 'Income')
    
    st.plotly_chart(fig)
    
    st.write("Pearson's correlation coefficient from the Scipy library can be used to understand if there is a relationshiop between these two variables")
    
    pearsons_code = '''
        import scipy.stats as stats
print(stats.pearsonr(data['Age'], data['Income']))
    '''
    st.write("The following code prints")
    st.code(pearsons_code, language = 'python')
    
    output = stats.pearsonr(data['Age'], data['Income'])
    
    output_code = output
    
    st.code(output, language = 'python')
    
    st.write("The Pearson's Correlation Coefficent is **:red[0.34]**, which implies that there is **very small** correlation between the variables 'Age' and 'Income' ")
    
    st.subheader("With trendline")
    scatter_data = data[['Age', 'Income']]
    fig = px.scatter(scatter_data, x = 'Age', y = 'Income', trendline = 'ols')
    
    st.plotly_chart(fig)
    
    st.write("The equation of the line is as follows: ")
    st.write("**Income = 1107.59 * Age + 81182.1**")
    
    st.header("Categorical vs Numerical")
    
    categorical_cols = data[['Sex', 'Marital status','Education', 'Occupation', 'Settlement size']]
    numerical_cols = data[['Age', 'Income']]
    
    for category in categorical_cols:
        for numerical in numerical_cols:
            st.write("<div style='text-align: center;'>" + category + " vs " + numerical + "</div>", unsafe_allow_html=True)
            fig = px.histogram(data, x=numerical, color=category, marginal='box')
            st.plotly_chart(fig)
    
    
    st.subheader("Categorical vs Categorical")
    
    
    fig = None
    for category1 in categorical_cols:
        cat_aux = [cat for cat in categorical_cols if cat != category1]
        st.subheader(category1)
        for category2 in cat_aux:
            fig = px.histogram(
                data,
                x=category1,
                color=category2,
                barmode='group'
            )
            st.write("<div style='text-align: center;'>" + category1 + " vs " + category2 + "</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            
    st.subheader("_Results/Conclusions_")
    
    
    points = [
        'Most customers living in small cities are unemployed (Occupation being 0).', 
        'Customer who are married tend to have higher levels of education.', 
        'Customer who are married also tend to live in smaller cities.',
        'Women in the dataset have more tendency towards being married, compared to man.',
        'Men in the dataset have more tendency towards being employed, while women in the dataset have more tendency towards being unemployed.'
    ]

    for point in points:
        st.markdown("- " + point)
    
        
elif page == 'Multivariate Analysis':
    
    if st.button("Take me to conclusion/insights!"):
        pass
    
    def bivariate_scatter(x, y, hue, data):
        
        # Create scatter plot with bright colors
        fig = px.scatter(
            data, x=x, y=y, color=hue, opacity=0.85,
            color_continuous_scale = 'electric'
        )
    
        
        fig.update_layout(
        plot_bgcolor='gray',                                                                        
        paper_bgcolor='gray'                                                                            
        )
        
        # Show plot
        return st.plotly_chart(fig, use_container_width=True)

        
    categorical_cols = data[['Sex', 'Marital status','Education', 'Occupation', 'Settlement size']]
    
    for cat in categorical_cols:
        st.markdown("**" + cat + "** in Age vs Income")
        bivariate_scatter('Age', 'Income', cat, data)
        

    
    st.subheader("_Results/Conclusions_")
        
    points = [
        'Unsurprisingly, people living in small cities have small incomes.', 
        'Again, unsurprisingly, people who have managerial jobs or people who own businesses have higher income than employees who also have higher income compared to unemployed people ',
        'People who are educated highly tend to be older but education levels seems to be irrelevant to the income.',
        'People who are non-single tend to have lower income, but they are somehown on average younger than single people, ',
        'When they are older, men tend to have higher income than women.'
    ]

    for point in points:
        st.markdown("- " + point)
elif page == "General Conclusion":
    points = [
        'There were no missing data',
        'Numerical values (Age and Income) seems to have right-skewed normal distribution, which is in need of transformation',
        'There is a small correlalation between Age and Income variables.',
        'People who live in small cities generally tend to have lower incomes',
        'The feature of Income get higher as the feature of Occupation gets higher in the hierarchy',
        'Non single people tend to have lower income compared to single people (This is surprising)',
        'And they also tend to be younger (This is even more surprising)',
        'When older, men tend to have higher income compared to women',
        'Most of the people who are unemployed and married live in small cities',
        'Women tend to be more unemployed compared to men'
    ]

    for point in points:
        st.markdown("- " + point)
else:
    feature_names = ['Normality Test for the Data', 'Income Feature', 'Age Feature', 'Scaling']
    feature_page = st.radio("", feature_names)
    divider()

    if feature_page == "Normality Test for the Data":
        st.write("In the \'Univariate Analysis\' section, the numeric features **Age** and **Income** showed a right-skewed distribution.")
        st.write("The K-means model needs data to be **normally distributed**")
        st.write("Hypothesis testing can be used to better understand normality of current features")
        divider()

        st.subheader("**H0 Null Hypothesis**: the features are normally distributed.")
        divider()
        st.subheader("**H1 Alternative Hypothesis**: the features are **:red[not]** normally distributed")
        divider()

        st.write("Scipy library can be used to conduct normality test")

        normality_test_code = """import scipy import stats      
stats.normaltest(customer_data['Income'])[1]
stats.normaltest(customer_data['Age'])[1]
        """

        st.code(normality_test_code)

        income_result = stats.normaltest(data['Income'])
        age_result = stats.normaltest(data['Age'])
        st.code(income_result)
        st.code(age_result)

        st.write("As it can be seen, the p-values for both feature to be normally distributed is extremely low")
        st.write("With high levels of certainity that it is lower than the generally accepted alpha level of **0.05**")
        st.subheader("Therefore, the **null hypothesis** is :red[rejected] in favor of the **alternative hypothesis**.")
        st.write("The features of Age and Income are not normally distributed and requires transformation")


    elif feature_page == "Income Feature":


        # PowerTransform data
        feature = data['Income'].to_numpy().reshape(-1,1)

        power_transformer = PowerTransformer()
        transformed_feature = power_transformer.fit_transform(feature)
        flattened_feature = transformed_feature.flatten()
        feature = pd.Series(data=flattened_feature, index=list(range(len(flattened_feature))))

        # Log Transform data
        log_transformed_income = np.log(data['Income'])

        # Create subplots for original data plot (ax1) and transformed data (ax2, ax3)
        fig = make_subplots(rows=1, cols=3, subplot_titles=['Original data', 'Log transformed data', 'PowerTransformed data'])

        fig.add_trace(go.Histogram(x=data['Income'], name='Original data'), row=1, col=1)

        norm_test1 = stats.normaltest(data['Income'])

        fig.add_trace(go.Histogram(x=log_transformed_income, name='Log transformed data'), row=1, col=2)

        norm_test2 = stats.normaltest(log_transformed_income)

        fig.add_trace(go.Histogram(x=feature, name='PowerTransformed data'), row=1, col=3)

        norm_test3 = stats.normaltest(feature)

        # Update layout and display the figure
        fig.update_layout(title='Distribution of Income data after transformations')
        st.plotly_chart(fig, use_container_width=True)

        # Create a DataFrame that shows normality test results for each tranformation
        norm_results = [(norm_test1.statistic, str(norm_test1.pvalue)), (norm_test2.statistic, str(norm_test2.pvalue)), (norm_test3.statistic, str(norm_test3.pvalue))]
        


        # create a pandas dataframe from the list
        df = pd.DataFrame(norm_results, columns=["Statistic", "P-value"]).T

        # display the dataframe in a streamlit page
        st.write(df.rename(columns = {0:'Original Data', 1: 'Log Transformation', 2:'Power Transformation'}))

        st.subheader("With lowest test statistic, :red[Power Transformation] seem to yield best result for :blue[Income Feature]")

    elif feature_page == "Age Feature":
        pass
        
        # PowerTransform data
        feature = data['Age'].to_numpy().reshape(-1,1)

        power_transformer = PowerTransformer()
        transformed_feature = power_transformer.fit_transform(feature)
        flattened_feature = transformed_feature.flatten()
        feature = pd.Series(data=flattened_feature, index=list(range(len(flattened_feature))))

        # Log Transform data
        log_transformed_age = np.log(data['Age'])

        # Create subplots for original data plot (ax1) and transformed data (ax2, ax3)
        fig = make_subplots(rows=1, cols=3, subplot_titles=['Original data', 'Log transformed data', 'PowerTransformed data'])

        fig.add_trace(go.Histogram(x=data['Age'], name='Original data', nbinsx = 12), row=1, col=1)

        norm_test1 = stats.normaltest(data['Age'])

        fig.add_trace(go.Histogram(x=log_transformed_age, name='Log transformed data', nbinsx = 12), row=1, col=2)

        norm_test2 = stats.normaltest(log_transformed_age)

        fig.add_trace(go.Histogram(x=feature, name='PowerTransformed data', nbinsx = 12), row=1, col=3)

        norm_test3 = stats.normaltest(feature)

        # Update layout and display the figure
        fig.update_layout(title='Distribution of Age data after transformations')
        st.plotly_chart(fig, use_container_width=True)

        # Create a DataFrame that shows normality test results for each tranformation
        norm_results = [(norm_test1.statistic, str(norm_test1.pvalue)), (norm_test2.statistic, str(norm_test2.pvalue)), (norm_test3.statistic, str(norm_test3.pvalue))]
        


        # create a pandas dataframe from the list
        df = pd.DataFrame(norm_results, columns=["Statistic", "P-value"]).T

        # display the dataframe in a streamlit page
        st.write(df.rename(columns = {0:'Original Data', 1: 'Log Transformation', 2:'Power Transformation'}))
        st.subheader("With lowest test statistic, :red[Log Transformation] seem to yield best result for :blue[Age Feature]")

    else:
        st.write("After deciding on which transformation techniques to use on numerical data, now its turn for scaling.")
        st.write("Scaling is necessary for K-means model to function more properly.")
        st.write("Having such type of a categorical data, its seems best to use MinMaxScaler")
        scaler_code = """from sklearn.preprocessing import MinMaxScaler"""
        st.code(scaler_code)
        st.subheader("Scaling will be done in the next step")















#################################################### hiding useless parts

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 