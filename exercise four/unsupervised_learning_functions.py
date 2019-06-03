import os
import pandas as pd
import pipeline as pl  # import the functions from our pipeline
import sklearn.preprocessing
import sklearn.tree
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans


def calculate_kmeans_clusters(data, number_of_clusters):
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(data)
    print('number of clusters is', number_of_clusters)

    return kmeans


def output_dummy_vars_by_cluster_with_lots_of_1s(data, mymodel_labels):
    print('This output will tell you which dummy variables in each cluster had a lot of 1s.\n' +
          'The idea is that this will help give a sense of which traits differentiate the clusters.')
    for cluster_label in np.unique(mymodel_labels):
        indices_in_cluster = mymodel_labels[
            np.where(mymodel_labels == cluster_label)]
        print('\n\n\n\n********************Cluster: ', cluster_label)
        description = data.iloc[indices_in_cluster].describe()
        print('variables with 75th percentile >=1:\n')
        for col in description.columns:
            if description[col]['75%'] >= 1:
                print(col)
        description.to_csv('figures/summary_cluster_number' +
                           str(cluster_label) + '.csv')


def make_barplot_of_variable_means_by_cluster(data, mymodel_labels):
    means = []
    for cluster_label in np.unique(mymodel_labels):
        indices_in_cluster = mymodel_labels[
            np.where(mymodel_labels == cluster_label)]
        current_cluster = data.iloc[indices_in_cluster]
        for col in current_cluster.columns:
            current_mean = current_cluster[col].mean()
            if current_mean > 0:
                means.append((cluster_label, col, current_mean))

    means = pd.DataFrame(means)
    means.columns = ['cluster', 'variable', 'mean']
    means['variable and cluster'] = means['variable'] + \
        "        clstr: " + means['cluster'].astype(str)
    plt.figure(figsize=(16, 60))
    sns.set(font_scale=1.5)
    g = sns.barplot(y="variable and cluster", x="mean", data=means)


def make_dotplot_of_variable_means_by_cluster(data, mymodel_labels):
    means = []
    for cluster_label in np.unique(mymodel_labels):
        indices_in_cluster = mymodel_labels[
            np.where(mymodel_labels == cluster_label)]
        current_cluster = data.iloc[indices_in_cluster]
        for col in current_cluster.columns:
            current_mean = current_cluster[col].mean()
            means.append((cluster_label, col, current_mean))

    means = pd.DataFrame(means)
    means.columns = ['cluster', 'variable', 'mean']
    means['variable and cluster'] = means['variable'] + \
        "        clstr: " + means['cluster'].astype(str)
    means['mean'] = means['mean'] + (
        (np.random.uniform(size=means.shape[0])*2 - 1) / 5)  # adding random
                                                             # jitter for plot
    df = means.pivot(index='variable',
                     columns='cluster',
                     values='mean').reset_index()
    # modified this code: https://python-graph-gallery.com/184-lollipop-plot-with-2-groups/
    num_clusters = means['cluster'].unique().shape[0]

    colors = ['red', 'blue', 'skyblue', 'hotpink', 'orange', 'purple',
              'green', 'yellow', 'black', 'grey', 'wheat', 'lavender',
              'magenta', 'darkblue', 'lime', 'forestgreen', 'navajowhite',
              'olivedrab', 'cyan', 'peru', 'azure', 'darkred']
    mycolors = colors[:num_clusters]
    ordered_df = df
    my_range = range(1, len(ordered_df.index)+1)

    plt.figure(figsize=(16, 60))
    plt.hlines(y=my_range, xmin=means['mean'].min(), xmax=means['mean'].max(),
               color='grey', alpha=0.1)
    for index, col in enumerate(ordered_df.columns[1:]):
        plt.scatter(ordered_df[col],
                    my_range,
                    color=mycolors[index],
                    label=str(col), s=75, alpha=.5)
    plt.legend()
    # Add title and axis names
    plt.yticks(my_range, ordered_df['variable'])
    plt.title(
        "Comparison of variable means by cluster, with random jitter added",
        loc='left')
    plt.xlabel('Means')
    plt.ylabel('Variable')


def get_tree_feature_importances_by_cluster(data, mymodel_labels):
    print("This gives feature importances for our various clusters:\n")
    for current_cluster in np.unique(mymodel_labels):
        cluster_label = current_cluster
        indices_in_cluster = mymodel_labels[
            np.where(mymodel_labels == cluster_label)]
        data_copy = data.copy()
        data_copy['cluster_id'] = mymodel_labels
        data_copy['in_cluster'] = 0
        data_copy.loc[data_copy['cluster_id'] == cluster_label,
                      'in_cluster'] = 1
        mymodel = sklearn.tree.DecisionTreeClassifier(max_depth=10)
        x_vars = data_copy.drop(['cluster_id', 'in_cluster'], axis=1)
        mymodel.fit(X=x_vars, y=data_copy['in_cluster'])
        list_of_vars_and_importances = list(
            zip(x_vars.columns, mymodel.feature_importances_))
        importance_df = pd.DataFrame(list_of_vars_and_importances,
                                     columns=['var', 'importance'])
        print('current cluster is ', cluster_label)
        print(importance_df.sort_values('importance', ascending=False).head())
        print('\n\n\n')


def create_kmeans_clusters_within_cluster(data,
                                          cluster_we_will_cluster_within,
                                          number_of_new_clusters,
                                          current_model):
    cluster_label = cluster_we_will_cluster_within
    indices_in_cluster = current_model.labels_[
        np.where(current_model.labels_ == cluster_label)]
    data_copy = data.copy()
    data_copy['cluster_id'] = current_model.labels_
    data_subset = data_copy.loc[data_copy['cluster_id'] == cluster_label, ]
    kmeans = KMeans(n_clusters=number_of_new_clusters)
    kmeans.fit(data_subset)

    return (kmeans, data_subset)


def redo_analysis_with_merged_cluster(data,
                                      mymodel_labels,
                                      list_of_clusters_to_merge):
    data_copy = data.copy()
    print('data copied')
    data_copy['cluster_id'] = mymodel_labels
    print('labels changed on data_copy')
    first_cluster_id_in_merge = list_of_clusters_to_merge[0]
    print('first_cluster id created')
    other_values = list_of_clusters_to_merge[1:]
    print('other values created')

    for cluster_id in other_values:
        data_copy.loc[
            data_copy['cluster_id'] == cluster_id,
            ['cluster_id']] = first_cluster_id_in_merge

    print('renamed clusters')

    mymodel_labels_revised = np.asarray(data_copy['cluster_id'])

    output_dummy_vars_by_cluster_with_lots_of_1s(data, mymodel_labels_revised)
    get_tree_feature_importances_by_cluster(data=data_copy,
                                            mymodel_labels=mymodel_labels_revised)
    make_dotplot_of_variable_means_by_cluster(data, mymodel_labels_revised)
    make_barplot_of_variable_means_by_cluster(data, mymodel_labels_revised)
