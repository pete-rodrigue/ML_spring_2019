'''
Machine learning HW 2
Spring 2019
pete rodrigue
'''

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np


def load_and_peek_at_data(path, summary=False):
    '''loads our data and returns a pandas dataframe'''
    df = pd.read_csv(path)
    print('Head of data:')
    print(df.head(5))
    print('Tail of data:')
    print(df.tail(5))
    print('column names of data:')
    print(df.columns)
    print('number of rows of data:')
    print(len(df))

    if summary:
        print("\n\n\nSummary of data:")
        print(df.describe())
        df.describe().to_csv('exercise two/figures/summary.csv')

    return df


def make_graphs(df):
    ''' lkdjfdlfkjdfl'''
    df_temp = df._get_numeric_data()
    for col in df_temp.columns:
        plt.clf()
        mycol = df_temp[col][df_temp[col].notna()]
        print('skew', ' for col ', mycol.name, 'is:', mycol.skew())
        if abs(mycol.skew()) > 10:
            path = "exercise two/figures/" + col + "log_transformed"
            g = sns.distplot(mycol)
            g.set_title(col + " dist, log_transformed")
            g.set(xscale='log')
            plt.savefig(path)
        else:
            path = "exercise two/figures/" + col
            g = sns.distplot(mycol)
            g.set_title(col + " distribution")
            plt.savefig(path)
        plt.clf()
        path = "exercise two/figures/" + col + " boxplot"
        g = sns.boxplot(mycol)
        plt.savefig(path)


def fill_missing(df):
    ''' lkdjfdlfkjdfl'''
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)


def descretize_var(df, var, num_groups):
    ''' lkdjfdlfkjdfl'''
    labs = list(range(1,num_groups + 1))
    labs = [str(x) for x in labs]
    new_var = var + '_discrete'
    df[new_var] = pd.qcut(df[var], num_groups, labels=labs)
    return df



# 1. Read/Load Data

df = load_and_peek_at_data('exercise two/credit-data.csv', summary=True)

# 2. Explore Data
make_graphs(df)

# 3. Pre-Process and Clean Data

fill_missing(df)

# 4. Generate Features/Predictors

# one function that can discretize a continuous variable
df = descretize_var(df, 'MonthlyIncome', 3)

# one function that can take a categorical
# variable and create binary/dummy variables from it.



# 5. Build Machine Learning Classifier

# 6. Evaluate Classifier
