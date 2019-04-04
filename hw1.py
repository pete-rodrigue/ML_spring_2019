'''
Machine learning HW 1
Spring 2019
pete rodrigue
'''

import pandas as pd


# Part one

# Uncomment the code below if you need to download the data again.

for myoffset in range(0, 50):
    print('myoffset is', myoffset)
    url_2017 = 'https://data.cityofchicago.org/resource/crimes.json?year=2017&$limit=50000&$offset=' + str(myoffset)
    url_2018 = 'https://data.cityofchicago.org/resource/crimes.json?year=2018&$limit=50000&$offset=' + str(myoffset)
    print(url_2017)
    results_2017 = pd.read_json(url_2017)
    results_2018 = pd.read_json(url_2018)
    if results_2017.empty:
        break
    if myoffset == 0:
        results_df = results_2017
        results_df = results_df.append(results_2018, ignore_index=True)
    else:
        results_df = results_df.append(results_2017, ignore_index=True)
        results_df = results_df.append(results_2018, ignore_index=True)

results_df.to_csv('alleged_crimes_data.csv', index=False)


df = pd.read_csv('alleged_crimes_data.csv')

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
