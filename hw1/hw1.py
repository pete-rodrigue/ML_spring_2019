'''
Machine learning HW 1
Spring 2019
pete rodrigue
'''

import pandas as pd
import seaborn as sns

# Part one

# Uncomment the code below if you need to download the data again.

'''
for index in range(0, 500000, 50000):
    print('index is', index)
    url_2017 = 'https://data.cityofchicago.org/resource/crimes.json?year=2017&$limit=50000&$offset=' + str(index)
    url_2018 = 'https://data.cityofchicago.org/resource/crimes.json?year=2018&$limit=50000&$offset=' + str(index)
    print('url_2017 is', url_2017)
    results_2017 = pd.read_json(url_2017)
    results_2018 = pd.read_json(url_2018)
    if results_2017.empty:
        break
    if index == 0:
        df_2017 = results_2017
        df_2018 = results_2018
    else:
        df_2017 = df_2017.append(results_2017, ignore_index=True)
        df_2018 = df_2018.append(results_2018, ignore_index=True)

df_2017.to_csv('alleged_crimes_2017.csv', index=False)
df_2018.to_csv('alleged_crimes_2018.csv', index=False)
'''


df_2017 = pd.read_csv('alleged_crimes_2017.csv')
df_2018 = pd.read_csv('alleged_crimes_2018.csv')

# Number of crimes of each type
grouped_by_crime_2017 = df_2017.groupby(
    'primary_type').size().sort_values(ascending=False)
grouped_by_crime_2017 = grouped_by_crime_2017.reset_index()
grouped_by_crime_2017.columns = ['type of crime', 'count']
# print(grouped_by_crime_2017.columns)
grouped_by_crime_2017['year'] = 2017
# print(grouped_by_crime_2017)

grouped_by_crime_2018 = df_2018.groupby(
    'primary_type').size().sort_values(ascending=False)
grouped_by_crime_2018 = grouped_by_crime_2018.reset_index()
grouped_by_crime_2018.columns = ['type of crime', 'count']
# print(grouped_by_crime_2018.columns)
grouped_by_crime_2018['year'] = 2018
print(grouped_by_crime_2018)

df = grouped_by_crime_2017.append(grouped_by_crime_2018, ignore_index=True)

# How they change over time
a = sns.catplot(y="type of crime", x="count", hue="year",
                data=df.loc[df['count'] >= 2500, :],
                height=6, kind="bar", palette="muted")
a.set_xlabels("Number of reported crimes")
axes_a = a.axes.flatten()
axes_a[0].set_title('Common crimes (at least 2,500 reported instances)')

b = sns.catplot(y="type of crime", x="count", hue="year",
                data=df.loc[df['count'] < 2500, :],
                height=6, kind="bar", palette="muted")
b.set_xlabels("Number of reported crimes")
axes_b = b.axes.flatten()
axes_b[0].set_title('Less common crimes (less than 2,500 reported instances)')

# How they are different by neighborhood
