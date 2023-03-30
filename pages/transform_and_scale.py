from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

data = pd.read_csv('./segmentation_data.csv')



class Income:
    def __init__(self, income):
        self.income = income
        print("Income value initialized as:", income)
        self.power_transformer = PowerTransformer()
        self.income_transformer = self.power_transformer.fit(data['Income'].to_numpy().reshape(-1, 1))
        self.transformed_data = self.income_transformer.transform(data['Income'].to_numpy().reshape(-1, 1))
        self.scaler = MinMaxScaler()
        print("TYPE:", type(self.transformed_data))
        self.scaled_income = self.scaler.fit(self.transformed_data.reshape(-1, 1))
    
    def transform_and_scale(self):
        value =  self.income_transformer.transform([[self.income]])
        print("Income value transformed as:", value)
        value = self.scaled_income.transform(value)
        print("Income value scaled as:", value)
        return float(value)



class Age:
    def __init__(self, age):
        self.age = age
        print("Age value initialized as: ", age)
        self.transformed_age = np.log(data['Age'].to_numpy().reshape(-1, 1))
        self.scaler = MinMaxScaler()
        self.scaled_age = self.scaler.fit(self.transformed_age)
    
    def transform_and_scale(self):
        value = np.log(self.age)
        print("Age value transformed as:", value)
        value = self.scaled_age.transform([[value]])
        print("Age value scaled as:", value)
        print("TYPE:", type(value))
        return float(value)
