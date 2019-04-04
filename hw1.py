'''
Machine learning HW 1
Spring 2019
pete rodrigue
'''

import pandas as pd


# Part one

# Uncomment the code below if you need to download the data again.

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


df = pd.read_csv('alleged_crimes_2018.csv')

# Column names:
# 'arrest', 'beat', 'block', 'case_number', 'community_area', 'date',
#        'description', 'district', 'domestic', 'fbi_code', 'id', 'iucr',
#        'latitude', 'location', 'location_description', 'longitude',
#        'primary_type', 'updated_on', 'ward', 'x_coordinate', 'y_coordinate',
#        'year'

# Number of crimes of each type
print(df.columns)
grouped_by_crime = df.groupby('primary_type').size().sort_values(ascending=False)
print(grouped_by_crime)
# How they change over time

# How they are different by neighborhood
