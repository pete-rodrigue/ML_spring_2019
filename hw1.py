'''
Machine learning HW 1
Spring 2019
pete rodrigue
'''

import pandas as pd


# Part one

results_2017 = pd.read_json(
    'https://data.cityofchicago.org/resource/crimes.json?year=2017')
results_2018 = pd.read_json(
    'https://data.cityofchicago.org/resource/crimes.json?year=2018')

results_df = results_2017.append(results_2018, ignore_index=True)
results_df.to_csv('alleged_crimes_data.csv', index=False)

df = pd.read_csv('alleged_crimes_data.csv')

# Number of crimes of each type

# How they change over time

# How they are different by neighborhood
