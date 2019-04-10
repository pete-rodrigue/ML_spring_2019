'''
Machine learning HW 1
Spring 2019
pete rodrigue
'''

import pandas as pd
import seaborn as sns
import json
import folium
import matplotlib.dates
import matplotlib.pyplot as plt
import datetime
import os

# You will need to set this to the root folder for the files
# os.chdir("C:/Users/edwar.WJM-SONYLAPTOP/Documents/GitHub/ML_spring_2019")

###############################
###############################
###############################
###############################
# Problem 1
###############################
###############################
###############################
###############################

########
# Part 1

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

df_2017.to_csv('figures/alleged_crimes_2017.csv', index=False)
df_2018.to_csv('figures/alleged_crimes_2018.csv', index=False)
'''

df_2017 = pd.read_csv('alleged_crimes_2017.csv')
df_2018 = pd.read_csv('alleged_crimes_2018.csv')

df_2017.size
df_2018.size

# Number of crimes of each type
grouped_by_crime_2017 = df_2017.groupby(
    'primary_type').size().sort_values(ascending=False)
grouped_by_crime_2017 = grouped_by_crime_2017.reset_index()
grouped_by_crime_2017.columns = ['type of crime', 'count']
grouped_by_crime_2017['year'] = 2017


grouped_by_crime_2018 = df_2018.groupby(
    'primary_type').size().sort_values(ascending=False)
grouped_by_crime_2018 = grouped_by_crime_2018.reset_index()
grouped_by_crime_2018.columns = ['type of crime', 'count']
grouped_by_crime_2018['year'] = 2018

df = grouped_by_crime_2017.append(grouped_by_crime_2018, ignore_index=True)
df.pivot(index='type of crime', columns='year', values='count').reset_index(
        ).sort_values(2017, ascending=False)

a = sns.catplot(y="type of crime", x="count", hue="year",
                data=df.loc[df['count'] >= 2500, :],
                height=6, kind="bar", palette="muted")
a.set_xlabels("Number of reported crimes")
axes_a = a.axes.flatten()
axes_a[0].set_title('Common crimes (at least 2,500 reported instances)')
a.savefig("figures/common crimes.png")
plt.close()

b = sns.catplot(y="type of crime", x="count", hue="year",
                data=df.loc[df['count'] < 2500, :],
                height=6, kind="bar", palette="muted")
b.set_xlabels("Number of reported crimes")
axes_b = b.axes.flatten()
axes_b[0].set_title('Less common crimes (less than 2,500 reported instances)')
b.savefig("figures/not common crimes.png")
plt.close()

# How they are different by neighborhood
df_2017 = pd.read_csv('alleged_crimes_2017.csv')
df_2018 = pd.read_csv('alleged_crimes_2018.csv')

df_all = df_2017.append(df_2018, ignore_index=True)

df = df_2018.groupby(['year', 'community_area', 'primary_type']).size(
                     ).reset_index()
df_all = df_all.groupby(['year', 'community_area', 'primary_type']).size(
                     ).reset_index()
df.columns = ['year', 'community_area', 'type of crime', 'count']
df_all.columns = ['year', 'community_area', 'type of crime', 'count']

community_area_names = pd.read_csv('CommAreas.csv')
community_area_names = community_area_names[['AREA_NUMBE', 'COMMUNITY']]
df = df.merge(
    community_area_names,
    how='left', left_on='community_area', right_on='AREA_NUMBE')
df = df.dropna(subset=['AREA_NUMBE'])

df_all = df_all.merge(
    community_area_names,
    how='left', left_on='community_area', right_on='AREA_NUMBE')
df_all = df_all.dropna(subset=['AREA_NUMBE'])

df_all = df_all.groupby(['year',
                         'COMMUNITY',
                         'type of crime']).sum().unstack('year')[
                'count'].reset_index()
df_all.fillna(0, inplace=True)
df_all['diff'] = df_all[2018] - df_all[2017]
df_all.sort_values('diff')
df_all.sort_values('diff').loc[df_all['type of crime'] == 'HOMICIDE']

g = sns.distplot(df_all.loc[df_all['type of crime'] == 'THEFT', 'diff'])
plt.axvline(df_all.loc[df_all['type of crime'] == 'THEFT', 'diff'].mean(),
            color='k', linestyle='dashed', linewidth=1)
fig = plt.gcf()
fig.savefig('figures/change in thefts.png')
plt.close()

sns.distplot(df_all.loc[df_all['type of crime'] == 'BATTERY', 'diff'])
plt.axvline(df_all.loc[df_all['type of crime'] == 'BATTERY', 'diff'].mean(),
            color='k', linestyle='dashed', linewidth=1)
fig = plt.gcf()
fig.savefig('figures/change in battery.png')
plt.close()

sns.distplot(df_all.loc[df_all['type of crime'] == 'HOMICIDE', 'diff'])
plt.axvline(df_all.loc[df_all['type of crime'] == 'HOMICIDE', 'diff'].mean(),
            color='k', linestyle='dashed', linewidth=1)
fig = plt.gcf()
fig.savefig('figures/change in homicides.png')
plt.close()

grouped_df = df.groupby(['COMMUNITY', 'type of crime']).sum().reset_index(
                        )[['COMMUNITY', 'type of crime', 'count']].sort_values(
                            'count', ascending=False)
wide_df = grouped_df.pivot(
    index='COMMUNITY', columns='type of crime', values='count').reset_index()
wide_df.fillna(0)

g = sns.PairGrid(wide_df.sort_values("THEFT", ascending=False),
                 x_vars=['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT'],
                 y_vars=["COMMUNITY"],
                 height=10, aspect=.25)

g.map(sns.stripplot, size=10, orient="h",
      palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")

g.set(xlabel="# reported crimes", ylabel="")

titles = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT']

for ax, title in zip(g.axes.flat, titles):

    ax.set(title=title)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
plt.gcf().set_size_inches(9, 18)
g.savefig('figures/reported crimes by community area.png')
plt.close()

# mapping with folium
chi_map = folium.Map(
    location=[41.860909, -87.630780],
    zoom_start=10)
folium.TileLayer('cartodbpositron', overlay=True).add_to(chi_map)
# creation of the choropleth
with open("Boundaries - Community Areas (current) (1).geojson") as f:
    geodata = json.load(f)

folium.Choropleth(
    geo_data=geodata,
    name='thefts',
    data=df.loc[df['type of crime'] == 'THEFT', :],
    columns=['COMMUNITY', 'count'],
    key_on='properties.community',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    overlay=False,
    legend_name='Number of reported thefts').add_to(chi_map)

folium.Choropleth(
    geo_data=geodata,
    name='battery',
    data=df.loc[df['type of crime'] == 'BATTERY', :],
    columns=['COMMUNITY', 'count'],
    key_on='properties.community',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    overlay=False,
    legend_name='Number of reported batteries').add_to(chi_map)

folium.Choropleth(
    geo_data=geodata,
    name='criminal damage',
    data=df.loc[df['type of crime'] == 'CRIMINAL DAMAGE', :],
    columns=['COMMUNITY', 'count'],
    key_on='properties.community',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    overlay=False,
    legend_name='Number of criminal damage reports').add_to(chi_map)

folium.Choropleth(
    geo_data=geodata,
    name='assault',
    data=df.loc[df['type of crime'] == 'ASSAULT', :],
    columns=['COMMUNITY', 'count'],
    key_on='properties.community',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    overlay=False,
    legend_name='Number of assault reports').add_to(chi_map)

folium.LayerControl('bottomleft').add_to(chi_map)

chi_map.save('figures/map.html')

# Timeseries of crime over last two years
df_2017 = pd.read_csv('alleged_crimes_2017.csv')
df_2018 = pd.read_csv('alleged_crimes_2018.csv')
df = df_2017.append(df_2018, ignore_index=True)

df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
timeseries = df.groupby(pd.Grouper(freq='M')).count()['arrest'].reset_index()
timeseries.columns = ['date', 'number of reported crimes']
timeseries['date'] = timeseries['date'].astype('str')
timeseries['date'] = matplotlib.dates.datestr2num(timeseries['date'])
timeseries.head()
timeseries.dtypes

g = sns.lineplot(x="date", y="number of reported crimes", data=timeseries)
fig = g.get_figure()
fig.savefig('figures/timeseries.png')
plt.close()

########
# Part 2

# Census data from social explorer:
# https://www.socialexplorer.com/tables/ACS2017_5yr/R12107097
# data dictionary is called R12107097_SL140.txt

census_data = pd.read_csv('R12107097_SL140.csv')
census_data.rename(columns={'SE_A10003_001': 'avgHHsize',
                            'SE_A14010_001': 'medianFamInc',
                            'SE_A13003A_001': 'totalPopUnder18',
                            'SE_A13003A_002': 'popUnder18InPov'}, inplace=True)

census_data = census_data[['Geo_FIPS', 'Geo_GEOID',
                           'avgHHsize', 'medianFamInc',
                           'totalPopUnder18', 'popUnder18InPov']]

census_data['shareInPov'] = census_data[
    'popUnder18InPov'] / census_data['totalPopUnder18']

# I conducted the spatial join of the reports to tracts in R.
# The code file that does that is called spatialJoinCrimesTracts.R
df_2018 = pd.read_csv('alleged_crimes_2018_with_tracts.csv')

census_data.Geo_GEOID = census_data.Geo_GEOID.str.findall(pat="(?<=US).*$")
census_data['Geo_GEOID'] = census_data['Geo_GEOID'].apply(lambda x: x[0])
census_data.Geo_GEOID = census_data.Geo_GEOID.astype('float64')
merged_2018 = pd.merge(df_2018, census_data,
                       left_on='geoid10', right_on='Geo_GEOID',
                       how='left')

df_2017 = pd.read_csv('alleged_crimes_2017_with_tracts.csv')

merged_2017 = pd.merge(df_2017, census_data,
                       left_on='geoid10', right_on='Geo_GEOID',
                       how='left')

merged_2018.loc[
    merged_2018['primary_type'] == 'BATTERY', "medianFamInc"].mean()
census_data.medianFamInc.mean()
merged_2018['medianFamInc'].fillna(0, inplace=True)
census_data['medianFamInc'].fillna(0, inplace=True)
merged_2018['shareInPov'].fillna(0, inplace=True)
census_data['shareInPov'].fillna(0, inplace=True)
merged_2018['avgHHsize'].fillna(0, inplace=True)
census_data['avgHHsize'].fillna(0, inplace=True)

merged_2017['medianFamInc'].fillna(0, inplace=True)
merged_2017['shareInPov'].fillna(0, inplace=True)
merged_2017['avgHHsize'].fillna(0, inplace=True)


def plot_scatter(outcome, crime):
    temp_df = merged_2018.loc[merged_2018['primary_type'] == crime,
                                         ['geoid10', outcome]].groupby(
                                         'geoid10').count().reset_index()
    varname = 'count of ' + crime + ' reports'
    temp_df.columns = ['geoid10', varname]
    temp_df = pd.merge(temp_df,
                       merged_2018.loc[merged_2018.primary_type == crime],
                       on='geoid10', how='left')
    temp_df = temp_df[[varname, outcome]]
    sns.lmplot(x=outcome, y=varname, data=temp_df,
               scatter_kws={'s': 8},
               lowess=True)
    fig = plt.gcf()
    path = 'figures/' + crime + '_' + outcome + '.png'
    fig.savefig(path)
    plt.close()


plot_scatter('medianFamInc', 'BATTERY')
plot_scatter('avgHHsize', 'BATTERY')
plot_scatter('shareInPov', 'BATTERY')

plot_scatter('medianFamInc', 'HOMICIDE')
plot_scatter('avgHHsize', 'HOMICIDE')
plot_scatter('shareInPov', 'HOMICIDE')

both_years = merged_2017.append(merged_2018, ignore_index=True)
both_years.head()
both_years.date = pd.to_datetime(both_years.date)
both_years.year = both_years.date.dt.year


def plot_multi_year(outcome, crime):
    temp_df = both_years.loc[both_years['primary_type'] == crime,
                                       ['geoid10', 'year', outcome]].groupby(
                                        ['year',
                                         'geoid10']).count().reset_index()
    varname = 'count of ' + crime + ' reports'
    temp_df.columns = ['year', 'geoid10', varname]
    temp_df = pd.merge(temp_df,
                       both_years.loc[both_years.primary_type == crime],
                       on='geoid10', how='left')
    temp_df = temp_df[['year_x', varname, outcome]]
    temp_df.rename(columns={'year_x': 'year'}, inplace=True)
    sns.lmplot(x=outcome, y=varname, data=temp_df,
               col='year',
               scatter_kws={'s': 6, 'alpha': 0.3},
               lowess=True)
    fig = plt.gcf()
    path = 'figures/' + 'both_years_' + crime + '_' + outcome + '.png'
    fig.savefig(path)
    plt.close()


plot_multi_year('medianFamInc', 'BATTERY')
plot_multi_year('avgHHsize', 'BATTERY')
plot_multi_year('shareInPov', 'BATTERY')

plot_multi_year('medianFamInc', 'HOMICIDE')
plot_multi_year('avgHHsize', 'HOMICIDE')
plot_multi_year('shareInPov', 'HOMICIDE')


def plot_two_vars(outcome, crime1, crime2):
    temp_df = merged_2018.loc[(merged_2018['primary_type'] == crime1) |
                              (merged_2018['primary_type'] == crime2),
                              ['geoid10',
                               'primary_type',
                               outcome]].groupby(
                                         ['primary_type',
                                          'geoid10']).count().reset_index()
    varname = 'count of reports'
    temp_df.columns = ['type of report', 'geoid10', varname]
    temp_df = pd.merge(temp_df,
                       merged_2018.loc[
                                (merged_2018['primary_type'] == crime1) |
                                (merged_2018['primary_type'] == crime2),
                                ['geoid10', 'tractce10', outcome]],
                       on='geoid10', how='left')
    temp_df = temp_df[['type of report', varname, outcome]]
    temp_df = temp_df.drop_duplicates()
    sns.scatterplot(x=outcome, y=varname, data=temp_df,
                    hue='type of report', s=6, alpha=0.5)
    fig = plt.gcf()
    path = 'figures/' + 'compare' + \
        crime1 + '_' + crime2 + '_' + outcome + '.png'
    fig.savefig(path)
    plt.close()
    temp_df['weighted_amount'] = temp_df[varname] * temp_df[outcome]
    total_num_reports_crime1 = temp_df.loc[
                    temp_df['type of report'] == crime1, varname].sum()
    total_num_reports_crime2 = temp_df.loc[
                    temp_df['type of report'] == crime2, varname].sum()
    weighted_avg_crime1 = temp_df.loc[
                    temp_df['type of report'] == crime1,
                    'weighted_amount'].sum() / total_num_reports_crime1
    weighted_avg_crime2 = temp_df.loc[
                    temp_df['type of report'] == crime2,
                    'weighted_amount'].sum() / total_num_reports_crime2
    print('weighted average for outcome: ',
          outcome, " and crime: ", crime1, "is ", weighted_avg_crime1, '\n',
          'weighted average for outcome: ',
          outcome, " and crime: ", crime2, "is ", weighted_avg_crime2, '\n')


plot_two_vars('medianFamInc', 'DECEPTIVE PRACTICE', 'SEX OFFENSE')
plot_two_vars('avgHHsize', 'DECEPTIVE PRACTICE', 'SEX OFFENSE')
plot_two_vars('shareInPov', 'DECEPTIVE PRACTICE', 'SEX OFFENSE')


########
# Part 3

# Question 2

month_in_july_2017 = both_years[
    (both_years['date'] >= datetime.date(2017, 6, 26)) &
    (both_years['date'] <= datetime.date(2017, 7, 26))]

month_in_july_2018 = both_years[
    (both_years['date'] >= datetime.date(2018, 6, 26)) &
    (both_years['date'] <= datetime.date(2018, 7, 26))]

month_in_july_2017.primary_type.unique()
month_in_july_2017.ward.unique()

month_in_july_2017.loc[month_in_july_2017.primary_type=='ROBBERY',:].size
month_in_july_2018.loc[month_in_july_2018.primary_type=='ROBBERY',:].size

month_in_july_2017.loc[
    (month_in_july_2017.primary_type == 'ROBBERY') &
    (month_in_july_2017.ward == 43), :].size
month_in_july_2018.loc[
    (month_in_july_2018.primary_type == 'ROBBERY') &
    (month_in_july_2018.ward == 43), :].size

month_in_july_2017.loc[month_in_july_2017.primary_type=='BATTERY',:].size
month_in_july_2018.loc[month_in_july_2018.primary_type=='BATTERY',:].size

month_in_july_2017.loc[
    (month_in_july_2017.primary_type == 'BATTERY') &
    (month_in_july_2017.ward == 43), :].size
month_in_july_2018.loc[
    (month_in_july_2018.primary_type == 'BATTERY') &
    (month_in_july_2018.ward == 43), :].size

month_in_july_2017.loc[month_in_july_2017.primary_type=='BURGLARY',:].size
month_in_july_2018.loc[month_in_july_2018.primary_type=='BURGLARY',:].size

month_in_july_2017.loc[(month_in_july_2017.primary_type=='BURGLARY') &
    (month_in_july_2017.ward==43),:].size
month_in_july_2018.loc[(month_in_july_2018.primary_type=='BURGLARY') &
    (month_in_july_2018.ward==43),:].size

month_in_july_2017.loc[month_in_july_2017.primary_type=='MOTOR VEHICLE THEFT',:].size
month_in_july_2018.loc[month_in_july_2018.primary_type=='MOTOR VEHICLE THEFT',:].size

month_in_july_2017.loc[(month_in_july_2017.primary_type=='MOTOR VEHICLE THEFT') &
    (month_in_july_2017.ward==43),:].size
month_in_july_2018.loc[(month_in_july_2018.primary_type=='MOTOR VEHICLE THEFT') &
    (month_in_july_2018.ward==43),:].size


month_in_july_2017.size
month_in_july_2018.size
(186108 - 180882) / 186108

###############################
###############################
###############################
###############################
# Problem 4
###############################
###############################
###############################
###############################

########
# Part 1
both_years.loc[both_years.tractce10 == 330100, :].groupby(
    'primary_type').size().sort_values(
    ascending=False)
both_years.loc[both_years.tractce10 == 330100, :].groupby(
    'primary_type').size().sort_values(
        ascending=False) / both_years.loc[
        both_years.tractce10 == 330100, :].groupby('primary_type').size(
        ).sort_values(ascending=False).sum()

both_years.loc[both_years.commarea == 33, :]
both_years.commarea.unique()

########
# Part 2
# Garfield Park
both_years.loc[(both_years.commarea == 27) |
               (both_years.commarea == 26), :].groupby(
    'primary_type').size().sort_values(
    ascending=False)
both_years.loc[(both_years.commarea == 27) |
               (both_years.commarea == 26), :].groupby(
               'primary_type').size().sort_values(
               ascending=False) / both_years.loc[(both_years.commarea == 27) |
                                                 (both_years.commarea == 26),
                                                 :].groupby(
                                                 'primary_type').size(
        ).sort_values(ascending=False).sum()


# Uptown
both_years.loc[(both_years.commarea == 3), :].groupby(
    'primary_type').size().sort_values(
    ascending=False)
both_years.loc[(both_years.commarea == 3), :].groupby(
    'primary_type').size().sort_values(
        ascending=False) / both_years.loc[(
                    both_years.commarea == 3), :].groupby('primary_type').size(
        ).sort_values(ascending=False).sum()
