# scratchspace
import pandas as pd
from sklearn import tree
import os
import numpy as np

def fill_missing(df):
    '''
    Fill missing numerica data in our data frame with the median value of that
    variable. Modifies the dataframe in place. Does not return anything.

    Inputs:
        df (pandas dataframe): our dataframe we want to modify
    '''
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].mean()
            df[col].fillna(median_val, inplace=True)


# initialize list of lists
mydata = [['tom', 10, 120, 1], ['nick', 15, 180, 0], ['juli', 14, 200, 0], ['pedro', None, 120, 1]]
# Create the pandas DataFrame
# df = pd.DataFrame(mydata, columns=['Name', 'Age', 'Weight', 'Outcome'])

os.chdir('C:/Users/edwar.WJM-SONYLAPTOP/Downloads')
df = pd.read_csv('cereal.csv', sep=';', skiprows=[1], dtype={'calories':"float64", 'protein':"float64", 'fat':"float64", 'sodium':"float64", 'fiber':"float64",'carbo':"float64", 'sugars':"float64", 'potass':"float64", 'vitamins':"float64", 'shelf':"float64", 'weight':"float64", 'cups':"float64", 'rating':"float64"})

df.columns
fill_missing(df)
df.head(2)

df.dtypes
df[['rating', 'cups']].describe()
df['Outcome'] = 0
df.loc[df['rating'] > 40, 'Outcome'] = 1
df.head()
mymodel = tree.DecisionTreeClassifier(max_depth=2)

mymodel.fit(X=df[['calories', 'protein', 'fat', 'sodium', 'fiber',
       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight']], y=df['Outcome'])
predicted_probs = pd.DataFrame(mymodel.predict_proba(df[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight']]))

predicted_probs

np.percentile()
