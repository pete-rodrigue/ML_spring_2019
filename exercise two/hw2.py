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
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


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
    labs = list(range(1, num_groups + 1))
    labs = [str(x) for x in labs]
    new_var = var + '_discrete'
    df[new_var] = pd.qcut(df[var], num_groups, labels=labs)

    return df


def make_dummies(df, var):
    ''' lkdjfdlfkjdfl'''
    new_var_prefix = "D_" + var

    return pd.concat([df, pd.get_dummies(df[var], prefix=new_var_prefix)],
                     axis=1)


def run_tree_model(df, x_data, y_data, max_depth, outcome_labels):
    '''kdlkdlksdjfl'''
    model = tree.DecisionTreeClassifier(max_depth=5)
    model.fit(X=x_data, y=y_data)

    print('feature names:\n', x_data.columns)
    print('feature importances:\n', model.feature_importances_)
    print('score:\n', model.score(X=x_data, y=y_data))

    tree.export_graphviz(model,
                         out_file='exercise two/figures/tree.dot',
                         feature_names=x_data.columns,
                         class_names=outcome_labels,
                         filled=True)

# 1. Read/Load Data

df = load_and_peek_at_data('exercise two/credit-data.csv', summary=True)

# 2. Explore Data
make_graphs(df)

# 3. Pre-Process and Clean Data

fill_missing(df)

sns.heatmap(df[df.columns.difference(
             ['PersonID',
              'SeriousDlqin2yrs',
              'zipcode',
              'NumberOfTime60-89DaysPastDueNotWorse',
              'NumberOfTimes90DaysLate'])].corr())

# 4. Generate Features/Predictors

# one function that can discretize a continuous variable
df = descretize_var(df, 'MonthlyIncome', 3)

# one function that can take a categorical
# variable and create binary/dummy variables from it.
make_dummies(df, 'MonthlyIncome_discrete')


# 5. Build Machine Learning Classifier
run_tree_model(df,
               x_data=df[df.columns.difference(
                            ['PersonID',
                             'SeriousDlqin2yrs',
                             'zipcode',
                             'NumberOfTime60-89DaysPastDueNotWorse',
                             'NumberOfTimes90DaysLate'])],
               y_data=df['SeriousDlqin2yrs'],
               max_depth=5,
               outcome_labels=['Not Delinquent', 'Delinquent'])


logisticRegr = LogisticRegression()
logisticRegr.fit(df[df.columns.difference(
             ['PersonID',
              'SeriousDlqin2yrs',
              'zipcode',
              'NumberOfTime60-89DaysPastDueNotWorse',
              'NumberOfTimes90DaysLate'])],
             df['SeriousDlqin2yrs'])

predictions = logisticRegr.predict(
                df[df.columns.difference(
                             ['PersonID',
                              'SeriousDlqin2yrs',
                              'zipcode',
                              'NumberOfTime60-89DaysPastDueNotWorse',
                              'NumberOfTimes90DaysLate'])])

cm = metrics.confusion_matrix(df['SeriousDlqin2yrs'], predictions)
score = logisticRegr.score(
    df[df.columns.difference(
                 ['PersonID',
                  'SeriousDlqin2yrs',
                  'zipcode',
                  'NumberOfTime60-89DaysPastDueNotWorse',
                  'NumberOfTimes90DaysLate'])],
        df['SeriousDlqin2yrs']
)

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)


# 6. Evaluate Classifier

predictions = model.predict(x_data)
accuracy_score(y_true=y_data, y_pred=predictions)
