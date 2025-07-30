import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(data):
    X = data[['YearsExperience']]
    y = data['Salary']
    model = LinearRegression()
    model.fit(X, y)
    return model
