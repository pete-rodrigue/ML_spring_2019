#!/usr/bin/env python
# coding: utf-8

# # Machine learning pipeline
# Spring 2019,
# pete rodrigue

# ## Load packages
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pylab
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from IPython.core.pylabtools import figsize
import random


# ## Define pipeline functions

def load_and_peek_at_data(path, date_vars=None, summary=False):
    '''
    Loads our data and returns a pandas dataframe.
    This function also saves a csv file with descriptive statistics for all
    our variables to our figures folder.
    Returns the pandas dataframe created from the csv file
    '''
    separator = '************************\n************************\n\n'
    df = pd.read_csv(path)
    print(separator)
    print('Head of data:')
    print(df.head(5))
    print(separator)
    print('Tail of data:')
    print(df.tail(5))
    print(separator)
    print('column names of data:')
    print(df.columns)
    print(separator)
    print('number of rows of data:')
    print(len(df))
    print(separator)

    if date_vars:
        for var in date_vars:
            df[var] = pd.to_datetime(df[var])

    if summary:
        print("\n\n\nSummary of data:")
        print(df.describe())
        # this exports a set of summary statistics to a csv file
        # in the figures folder
        df.describe().to_csv('figures/summary.csv')

    return df


def make_graphs(df, normal_qq_plots=False):
    '''
    Takes our dataframe, fills in missing values with the median,
    and outputs a series of plots:
            - Normal qq plots for each variable
            - Boxplots for each variable
            - Histograms for each variable
        - A correlation plot for all our variables

    Inputs:
        df (pandas dataframe): our dataframe we want to modify
        normal_qq_plots (bool): whether you want normal qq plots
    '''
    # create a temporary dataframe that only has our numeric variables
    df_temp = df._get_numeric_data()
    # fill the missing values for now, just for ease of plotting.
    # be aware that this is filling the missing observations using values
    # from the entire dataset.
    # the distributions used to fill the missing values may look different
    # after you conduct
    # temporal splits.
    fill_missing(df_temp)
    # correlation plot:
    plt.figure(figsize=(8, 8))
    plt.tight_layout()
    g = sns.heatmap(df[df.columns.difference(
                 ['PersonID',
                  'SeriousDlqin2yrs',
                  'zipcode',
                  'NumberOfTime60-89DaysPastDueNotWorse',
                  'NumberOfTimes90DaysLate'])].corr())
    plt.savefig('figures/correlation_plot')  # export to figures folder
    plt.close()

    # this for loop will loop through each of our varaibles and export
    # distribution plots to the figures folder
    for col in df_temp.columns:
        mycol = df_temp[col][df_temp[col].notna()]
        print('skew', ' for col ', mycol.name, 'is:', mycol.skew())
        # if the skew of our distribution is greater than 10, log transform it
        if abs(mycol.skew()) > 10:
            path = "figures/" + col + "log_transformed"
            g = sns.distplot(mycol)
            g.set_title(col + " dist, log_transformed")
            g.set(xscale='log')
            # export our distribution plot to the figures folder
            plt.savefig(path)
            if normal_qq_plots:
                path = "figures/" + col + " normal_qq_plot log trans"
                g = stats.probplot(np.log(df[col]+.0001),
                                   dist="norm", plot=pylab)
                plt.title(col + " normal_qq log transformed")
                plt.savefig(path)
        else:
            # if the skew is not greater than 10, plot the raw distribution:
            path = "figures/" + col
            g = sns.distplot(mycol)
            g.set_title(col + " distribution")
            plt.savefig(path)
            plt.close()
            if normal_qq_plots:
                path = "figures/" + col + " normal_qq_plot"
                g = stats.probplot(df[col], dist="norm", plot=pylab)
                plt.title(col + " normal_qq")
                plt.savefig(path)

        # plot and export boxplots for all of our variables
        plt.clf()
        path = "figures/" + col + " boxplot"
        g = sns.boxplot(mycol)
        plt.savefig(path)


def fill_missing(df, imputation_method='mean'):
    '''
    Fill missing numerica data in our data frame with the median value of that
    variable. Modifies the dataframe in place. Does not return anything.

    Inputs:
        df (pandas dataframe): our dataframe we want to modify
        imputation_method (string): the imputation method to use.
                                    Your choices are 'median' and 'mean'.
                                    The default is mean.
    '''
    for col in df.columns:
        if df[col].isnull().values.any():
            if imputation_method == 'mean':
                imputed_val = df[col].mean()
            else:
                imputed_val = df[col].median()
            df[col].fillna(imputed_val, inplace=True)


def descretize_var(df, var, num_groups):
    '''
    Takes one of our variables and splits it into discrete groups.

    Inputs:
        df (pandas dataframe): our dataframe we want to modify
        var (str): the column in our dataframe that we want to make a
                   categorical variable from
        num_groups (int): the number of groups our discrete variable will have

    Returns: a modified dataframe.
    '''
    labs = list(range(1, num_groups + 1))  # Get the number of groups
    labs = [str(x) for x in labs]         # Get the labels of the groups
    new_var = var + '_discrete'           # create new discrete variable
    # populate that new variable
    df[new_var] = pd.qcut(df[var], num_groups, labels=labs)

    return df


def make_dummies(df, var):
    '''
    Takes our dataframe and turns a specified variable into a series of
    dummy columns. This function returns the modified dataframe.

    Inputs:
        df (pandas dataframe): our dataframe we want to modify
        var (str): the column in our dataframe that we want to make dummies of

    Returns: a modified dataframe. dummy variables will have the prefix "D_"
    '''
    new_var_prefix = "D_" + var

    return pd.concat([df, pd.get_dummies(df[var], prefix=new_var_prefix)],
                     axis=1)


def print_confusion_matrix(cm):
    '''prints a confusion matrix'''
    print('confusion matrix')
    print('|T neg, F pos|\n|F neg, T pos|')
    print(cm)


# ## Model functions. These will be combined below into one function,
# called "compare_models"

# ### Tree
def run_tree_model(x_data, y_data, x_test=None,
                   y_test=None, max_depth=5,
                   outcome_labels=None, threshold=50,
                   use_test_sets=False):
    '''
    This function takes our data and computes a decision tree model.
    It saves a .dot file you can open in graphviz to see the tree.
    Inputs:
        x_data (pandas dataframe): data frame where each column is a predictor
        y_data (pandas series): series of outcomes
        max_depth (int): the maximum depth of the tree.
        outcome_labels (list of str): the labels for our predictor variables.
    '''
    mymodel = tree.DecisionTreeClassifier(max_depth=max_depth)
    mymodel.fit(X=x_data, y=y_data)
    print("***************Tree model")

    if use_test_sets:
        print('Returning test set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_test))

        return predicted_probs
    else:
        print('Returning training set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_data))

        return predicted_probs


# ### Logit model
def run_logit_model(x_data, y_data, x_test=None, y_test=None,
                    threshold=50, use_test_sets=False):
    '''
    This function takes our x and y data and a threshold,
    and computes a logistic model. It exports a confusion matrix table.

    Inputs:
        x_data (pandas dataframe): data frame where each column is a predictor
        y_data (pandas series): series of outcomes
    '''
    mymodel = LogisticRegression()
    mymodel.fit(x_data, y_data)
    print('***********Logistic regression')

    if use_test_sets:
        print('Returning test set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_test))

        return predicted_probs
    else:
        print('Returning training set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_data))

        return predicted_probs


# ### K-Nearest Neighbor
def run_knn_model(x_data, y_data, x_test=None, y_test=None,
                  num_n=2, threshold=50, use_test_sets=False):
    '''
    This function takes our x and y data and a threshold,
    and computes a knn model. It exports a confusion matrix table.

    Inputs:
        x_data (pandas dataframe): data frame where each column is a predictor
        y_data (pandas series): series of outcomes
        num_n (int): the number of neighbors
    '''
    mymodel = KNeighborsClassifier(n_neighbors=num_n)
    mymodel.fit(x_data, y_data)
    print('************KNN')

    if use_test_sets:
        print('Returning test set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_test))

        return predicted_probs
    else:
        print('Returning training set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_data))

        return predicted_probs


# SVM
def run_svm_model(x_data_scaled, y_data, x_test=None,
                  y_test=None, kernel='linear',
                  threshold=50, use_test_sets=False,
                  my_svc_tol=.0000001,
                  my_svc_random_state=0,
                  my_svc_C=0.01):
    '''
    Runs an SVM model on your data
    Note: this will run much faster if you scale your x data first
    '''
    mymodel = svm.LinearSVC(tol=my_svc_tol,
                            random_state=my_svc_random_state,
                            C=my_svc_C)
    mymodel.fit(x_data_scaled, y_data)
    print('************SVC')

    if use_test_sets:
        print('Returning test set performance:')
        predicted_probs = pd.DataFrame(mymodel.decision_function(x_test))

        return predicted_probs
    else:
        print('Returning training set performance:')
        predicted_probs = pd.DataFrame(mymodel.decision_function(x_data_scaled))

        return predicted_probs


# Random Forest
def run_forest(x_data, y_data, x_test=None, y_test=None,
               my_n_estimators=10, my_max_depth=5,
               threshold=50, use_test_sets=False):
    '''
    Runs a random forest model
    Inputs:
        x_data (pd dataframe) : our predictor data
        y_data (pd dataframe) : our outcome data
        n_estimators (int): the number of trees in our forest
        max_depth (int): the max number of levels in our trees
    '''
    mymodel = RandomForestClassifier(n_estimators=my_n_estimators,
                                     max_depth=my_max_depth)
    mymodel.fit(x_data, y_data)

    print('************Random Forest')

    if use_test_sets:
        print('Returning test set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_test))

        return predicted_probs
    else:
        print('Returning training set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_data))

        return predicted_probs


# Boosting
def run_boosted_model(x_data, y_data, x_test=None, y_test=None,
                      my_max_depth=5, my_n_estimators=100,
                      threshold=50, use_test_sets=False):
    '''
    Run a boosted decision tree model
    '''
    mymodel = AdaBoostClassifier(tree.DecisionTreeClassifier(
                                max_depth=my_max_depth),
                                 algorithm="SAMME",
                                 n_estimators=my_n_estimators)
    mymodel.fit(x_data, y_data)
    print('************Boosted Decision Tree')

    if use_test_sets:
        print('Returning test set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_test))

        return predicted_probs

    else:
        print('Returning training set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_data))

        return predicted_probs


# Bagging
def run_bagging_model(x_data, y_data, x_test=None,
                      y_test=None, use_test_sets=False,
                      max_depth=20, my_max_bagging_samples=200):
    '''
    Runs a bagging model
    '''
    mymodel = BaggingClassifier(
                    tree.DecisionTreeClassifier(max_depth=max_depth),
                    max_samples=my_max_bagging_samples)
    mymodel.fit(x_data, y_data)
    print('************Bagged Tree')

    if use_test_sets:
        print('Returning test set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_test))

        return predicted_probs

    else:
        print('Returning training set performance:')
        predicted_probs = pd.DataFrame(mymodel.predict_proba(x_data))

        return predicted_probs


# ### Looping through models to compare performance
def compare_models(x_data=None, x_data_scaled=None, y_data=None,
                   x_test=None, x_test_scaled=None, y_test=None,
                   use_test_data=False,
                   run_bagging=False, run_boosted=False, run_a_forest=False,
                   run_svm=False, run_knn=False, run_logit=False,
                   run_tree=False,
                   mythresholds=[50],
                   my_max_depth=5, my_n_estimators=100,
                   num_n=16,
                   outcome_labels=None,
                   mykernel='linear',
                   my_max_bagging_samples=200,
                   my_svc_tol=.0000001,
                   my_svc_random_state=0,
                   my_svc_C=0.01,
                   split_name=""):
    '''
    Compare all our models
    '''
    model = []  # records the model we used
    fpr = []    # records the false positive rate we got
    tpr = []    # records the true positive rate we got
    precision = []   # records the precision we got
    current_threshold = []   # records teh threshold we used

    # in each of these if statements, we loop through the thresholds,
    # recalculating the model each time.
    if run_bagging:
        predicted_probs = run_bagging_model(x_data, y_data,
                                            x_test=x_test,
                                            y_test=y_test,
                                            use_test_sets=use_test_data,
                                            max_depth=my_max_depth)
        for t in mythresholds:
            print('Threshold: ', t)
            predicted_probs['predicted_class'] = 0
            # get cut_off to classify *threshold* percent of rows as positive
            cut_off = np.percentile(a=predicted_probs[1], q=t)
            predicted_probs.loc[predicted_probs[1] >= cut_off,
                                'predicted_class'] = 1
            cm = metrics.confusion_matrix(y_test,
                                          predicted_probs['predicted_class'])
            print_confusion_matrix(cm)
            tpr.append(cm[1][1] / sum(cm[1]))   # append true positive rate
            fpr.append(cm[0][1] / sum(cm[0]))   # append false positive rate
            model.append('bagging')
            current_threshold.append(t)
            precision.append(cm[1][1] / (cm[1][1] + cm[0][1]))
            print('\n')
    if run_boosted:
        predicted_probs = run_boosted_model(x_data=x_data, y_data=y_data,
                                            my_max_depth=my_max_depth,
                                            my_n_estimators=my_n_estimators,
                                            x_test=x_test, y_test=y_test,
                                            use_test_sets=use_test_data)
        for t in mythresholds:
            print('Threshold: ', t)
            predicted_probs['predicted_class'] = 0
            # get cut_off to classify *threshold* percent of rows as positive
            cut_off = np.percentile(a=predicted_probs[1], q=t)
            predicted_probs.loc[predicted_probs[1] >= cut_off,
                                'predicted_class'] = 1
            cm = metrics.confusion_matrix(y_test,
                                          predicted_probs['predicted_class'])
            print_confusion_matrix(cm)
            tpr.append(cm[1][1] / sum(cm[1]))
            fpr.append(cm[0][1] / sum(cm[0]))
            model.append('boosted')
            current_threshold.append(t)
            precision.append(cm[1][1] / (cm[1][1] + cm[0][1]))
            print('\n')
    if run_a_forest:
        predicted_probs = run_forest(x_data, y_data, x_test, y_test,
                                     my_n_estimators, my_max_depth,
                                     use_test_sets=use_test_data)
        for t in mythresholds:
            print('Threshold: ', t)
            predicted_probs['predicted_class'] = 0
            # get cut_off to classify *threshold* percent of rows as positive
            cut_off = np.percentile(a=predicted_probs[1], q=t)
            predicted_probs.loc[predicted_probs[1] >= cut_off,
                                'predicted_class'] = 1
            cm = metrics.confusion_matrix(y_test,
                                          predicted_probs['predicted_class'])
            print_confusion_matrix(cm)
            tpr.append(cm[1][1] / sum(cm[1]))
            fpr.append(cm[0][1] / sum(cm[0]))
            model.append('forest')
            current_threshold.append(t)
            precision.append(cm[1][1] / (cm[1][1] + cm[0][1]))
            print('\n')
    if run_svm:
        predicted_probs = run_svm_model(x_data_scaled, y_data,
                                        x_test=x_test_scaled,
                                        y_test=y_test,
                                        use_test_sets=use_test_data,
                                        kernel=mykernel,
                                        my_svc_tol=my_svc_tol,
                                        my_svc_random_state=my_svc_random_state,
                                        my_svc_C=my_svc_C)
        for t in mythresholds:
            print('Threshold: ', t)
            predicted_probs['predicted_class'] = 0
            # get cut_off to classify *threshold* percent of rows as positive
            cut_off = np.percentile(a=predicted_probs[0], q=t)
            predicted_probs.loc[predicted_probs[0] >= cut_off,
                                'predicted_class'] = 1
            cm = metrics.confusion_matrix(y_test,
                                          predicted_probs['predicted_class'])
            print_confusion_matrix(cm)
            tpr.append(cm[1][1] / sum(cm[1]))
            fpr.append(cm[0][1] / sum(cm[0]))
            model.append('svm')
            current_threshold.append(t)
            precision.append(cm[1][1] / (cm[1][1] + cm[0][1]))
            print('\n')
    if run_knn:
        predicted_probs = run_knn_model(x_data, y_data, num_n=num_n,
                                        x_test=x_test, y_test=y_test,
                                        use_test_sets=use_test_data)
        for t in mythresholds:
            print('Threshold: ', t)
            predicted_probs['predicted_class'] = 0
            # get cut_off to classify *threshold* percent of rows as positive
            cut_off = np.percentile(a=predicted_probs[1], q=t)
            predicted_probs.loc[predicted_probs[1] >= cut_off,
                                'predicted_class'] = 1
            cm = metrics.confusion_matrix(y_test,
                                          predicted_probs['predicted_class'])
            print_confusion_matrix(cm)
            tpr.append(cm[1][1] / sum(cm[1]))
            fpr.append(cm[0][1] / sum(cm[0]))
            model.append('knn')
            current_threshold.append(t)
            precision.append(cm[1][1] / (cm[1][1] + cm[0][1]))
            print('\n')
    if run_logit:
        predicted_probs = run_logit_model(x_data, y_data=y_data,
                                          x_test=x_test, y_test=y_test,
                                          use_test_sets=use_test_data)
        for t in mythresholds:
            print('Threshold: ', t)
            predicted_probs['predicted_class'] = 0
            # get cut_off to classify *threshold* percent of rows as positive
            cut_off = np.percentile(a=predicted_probs[1], q=t)
            predicted_probs.loc[predicted_probs[1] >= cut_off,
                                'predicted_class'] = 1
            cm = metrics.confusion_matrix(y_test,
                                          predicted_probs['predicted_class'])
            print_confusion_matrix(cm)
            tpr.append(cm[1][1] / sum(cm[1]))
            fpr.append(cm[0][1] / sum(cm[0]))
            model.append('logit')
            current_threshold.append(t)
            precision.append(cm[1][1] / (cm[1][1] + cm[0][1]))
            print('\n')
    if run_tree:
        print('Threshold: ', t)
        predicted_probs = run_tree_model(x_data,
                                         y_data=y_data,
                                         max_depth=my_max_depth,
                                         outcome_labels=outcome_labels,
                                         x_test=x_test, y_test=y_test,
                                         use_test_sets=use_test_data)
        for t in mythresholds:
            predicted_probs['predicted_class'] = 0
            # get cut_off to classify *threshold* percent of rows as positive
            cut_off = np.percentile(a=predicted_probs[1], q=t)
            predicted_probs.loc[predicted_probs[1] >= cut_off,
                                'predicted_class'] = 1
            cm = metrics.confusion_matrix(y_test,
                                          predicted_probs['predicted_class'])
            print_confusion_matrix(cm)
            tpr.append(cm[1][1] / sum(cm[1]))
            fpr.append(cm[0][1] / sum(cm[0]))
            model.append('tree')
            current_threshold.append(t)
            precision.append(cm[1][1] / (cm[1][1] + cm[0][1]))
            print('\n')

    # once our selected models have run, this piece of code creates a pandas
    # dataframe with our different model results.
    # We then export the table using a random number that identifies the
    # model run, and plot a line plot of the performance
    # (true positive rate over false positive rate)
    # We also print the AUC for each model.
    rows_to_add = len(current_threshold)
    to_plot = pd.DataFrame({'model': model + ['baseline'] * rows_to_add,
                            'tpr': tpr + fpr[0:rows_to_add],
                            'fpr': fpr + fpr[0:rows_to_add],
                            'precision': precision + [None] * rows_to_add,
                            'threshold': current_threshold + [None] * rows_to_add})
    print('\n\n')
    for m in to_plot['model'].unique():
        auc = -1*np.trapz(y=to_plot.loc[to_plot['model'] == m, 'tpr'],
                          x=to_plot.loc[to_plot['model'] == m, 'fpr'])
        print('AUC for ', m, ' is', auc)

    print('\n\nTable of results:')
    print(to_plot)
    to_plot.to_csv(split_name + ".csv")
    plt.clf()
    sns.lineplot(x=to_plot['fpr'],
                 y=to_plot['tpr'],
                 hue=to_plot['model'])
    plt.savefig('figures/fpr_vs_tpr' + split_name)


# ## Function to split data into test and training data
def split_using_date(data, train_start_date, train_end_date,
                     test_start_date, test_end_date):
    '''
    Splits our data into test and training sets using a date the user provides
    '''
    train = data.loc[data['date_posted'].between(
                                                    train_start_date,
                                                    train_end_date,
                                                    inclusive=True), :]
    test = data.loc[data['date_posted'].between(
                                                    test_start_date,
                                                    test_end_date,
                                                    inclusive=False), :]

    return [train, test]
