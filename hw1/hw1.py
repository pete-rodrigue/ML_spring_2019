'''
Machine learning HW 1
Spring 2019
pete rodrigue
'''

import pandas as pd
import seaborn as sns
import json
import folium
from bokeh.models import GeoJSONDataSource
from bokeh.models import GeoJSONDataSource, LogColorMapper
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN

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

print(df_2017.columns)

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
a.savefig("hw1/common crimes.png")

b = sns.catplot(y="type of crime", x="count", hue="year",
                data=df.loc[df['count'] < 2500, :],
                height=6, kind="bar", palette="muted")
b.set_xlabels("Number of reported crimes")
axes_b = b.axes.flatten()
axes_b[0].set_title('Less common crimes (less than 2,500 reported instances)')
b.savefig("hw1/not common crimes.png")

# How they are different by neighborhood
'''
df = df_2017.append(df_2018, ignore_index=True)
df = df.groupby(['year', 'community_area', 'primary_type']).size().reset_index()
df.columns = ['year', 'community_area', 'type of crime', 'count']

with open("hw1/Boundaries - Community Areas (current) (1).geojson") as f:
    geodata = json.load(f)

print(geodata['features'][0]['properties']['area_numbe'])

for idx in range(len(geodata['features'])):
    current_community_area = geodata['features'][idx]['properties']['area_numbe']
    print(current_community_area)
    geodata['features'][idx]['properties']['ASSAULT'] = df.loc[
        (df['community_area'] == int(current_community_area)) & \
        (df['year'] == 2018), ['type of crime', 'count']].set_index('type of crime').to_dict()['count']['ASSAULT']
    geodata['features'][idx]['properties']['THEFT'] = df.loc[
        (df['community_area'] == int(current_community_area)) & \
        (df['year'] == 2018), ['type of crime', 'count']].set_index('type of crime').to_dict()['count']['THEFT']


json_data = json.dumps(geodata)
# print(json_data)
geo_source = GeoJSONDataSource(geojson=json_data)

p = figure(title="Chicago")
color_column = 'ASSAULT'
p.patches('xs', 'ys', fill_alpha=0.7,
          fill_color={'field': color_column, 'transform': LogColorMapper(palette=palette)},
          line_color='black', line_width=0.5,
          source=geo_source
          )

outfile = open('map_of_neighborhoods.html', 'w')
outfile.write(file_html(p, CDN, 'reported crime'))
outfile.close()
'''






df_2017 = pd.read_csv('alleged_crimes_2017.csv')
df_2018 = pd.read_csv('alleged_crimes_2018.csv')

df_2017.head()
df = df_2017.groupby(['year', 'community_area', 'primary_type']).size().reset_index()
df.columns = ['year', 'community_area', 'type of crime', 'count']

community_area_names = pd.read_csv('hw1/CommAreas.csv')
community_area_names.head()
community_area_names = community_area_names[['AREA_NUMBE', 'COMMUNITY']]
df = df.merge(
    community_area_names,
    how='left', left_on='community_area', right_on='AREA_NUMBE')
df = df.dropna(subset=['AREA_NUMBE'])

grouped_df = df.groupby(['COMMUNITY', 'type of crime']).sum().reset_index()[['COMMUNITY', 'type of crime', 'count']].sort_values('count', ascending=False)
grouped_df.head()
wide_df = grouped_df.pivot(index='COMMUNITY', columns='type of crime', values='count').reset_index()
wide_df.fillna(0)
wide_df.head()


g = sns.PairGrid(wide_df.sort_values("THEFT", ascending=False),
                 x_vars=['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT'], y_vars=["COMMUNITY"],
                 height=10, aspect=.25)

g.map(sns.stripplot, size=10, orient="h",
      palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")

g.set(xlabel="# reported crimes", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'ASSAULT']

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
plt.gcf().set_size_inches(16, 8.27)
g.savefig('hw1/reported crimes by community area.png')




# mapping with folium
chi_map = folium.Map(location=[41.860909, -87.630780], tiles='Stamen Toner', zoom_start=10)
# creation of the choropleth
with open("hw1/Boundaries - Community Areas (current) (1).geojson") as f:
    geodata = json.load(f)
chi_map.geo_json(geo_path = district_geo,
              data_out = 'crimeagg.json',
              data = crimedata2,
              columns = ['District', 'Number'],
              key_on = 'feature.properties.DISTRICT',
              fill_color = 'YlOrRd',
              fill_opacity = 0.7,
              line_opacity = 0.2,
              legend_name = 'Number of incidents per district')

chi_map.save('hw1/map.html')
