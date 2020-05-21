#!/usr/bin/env python
# coding: utf-8

# # Best places to visit in Milan

# This report will give the information neccesary when someone visit Milan-Italia, in this case for area or city in milan wil give you the 5 most important commerce in every venue, and the distance between a selected areas of Milan.

# I´m going to use the data of postalcode of the cities of Milan from :http://es.gpspostcode.com/codigo-postal/italiana/milano/?tri=cp_desc, that i pass to my repository in Github in a file csv to make more easy the transformation.
# 
# i´m going to use Foursquare to consult the places with relevance en every city or venue in Milan.

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install beautifulsoup4')
get_ipython().system('{sys.executable} -m pip install requests')
get_ipython().system('{sys.executable} -m pip install geopy')
get_ipython().system('{sys.executable} -m pip install sklearn')
get_ipython().system('{sys.executable} -m pip install folium')
get_ipython().system('{sys.executable} -m pip install matplotlib')


# In[2]:


from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import requests
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[34]:


#información de http://es.gpspostcode.com/codigo-postal/italiana/milano/?tri=cp_desc

data = pd.read_csv('https://raw.githubusercontent.com/josh2250/github-example/master/cp_milan.csv')


# In[35]:


data.head()


# In[36]:


print("Dataframe shape: ", df.shape)


# In[37]:


address = 'Milan, Italia'
geolocator = Nominatim(user_agent="milan_explorer")
location = geolocator.geocode(address)
print('The geograpical coordinate of Milan are {}, {}.'.format(
    location.latitude, location.longitude))


# In[58]:


map_milan = folium.Map(
    location=[location.latitude, location.longitude],
    zoom_start=10,
   tiles='CartoDB dark_matter')


# In[61]:


for index, row in data.iterrows():

    folium.CircleMarker(location=(row["X"],
                                  row["Y"]),
                        radius= 5,
                        color='blue',
                        fill_color="#007849",
                        popup=row["ciudad"],
                        fill=True).add_to(map_milan)


# In[121]:


# Create dataframe with boroughs containing the term 'Toronto'
milan_data = data[data.lugar.str.contains('Milan')]
milan_data.reset_index(inplace=True, drop=True)
milan_data


# In[62]:


map_milan


# In[63]:


# data from square
CLIENT_ID = 'RR3ZMGRDZADKUZOODJ3KAUXVH0KOLPSQ4JTBSAQPC2Z5PTIU'
CLIENT_SECRET ='PVE2JCJVCSCFPWUB0S25SQQH5GBDSTKYM34G1QZZC5UVNHGM'
VERSION = '20190601'


# In[67]:


neighborhood_name = data.loc[0, "ciudad"]
neighborhood_latitude = data.loc[0, "X"]
neighborhood_longitude = data.loc[0, "Y"]
print("Latitude and longitude of the city '{}' are [{}, {}]".format(
    neighborhood_name, neighborhood_latitude, neighborhood_longitude))


# In[68]:


# Construct the URL para square
limit = 100
radius = 500
explore_url_prefix = 'https://api.foursquare.com/v2/venues/explore'
url = '{}?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    explore_url_prefix, CLIENT_ID, CLIENT_SECRET, VERSION, 
    neighborhood_latitude, neighborhood_longitude, radius, limit)


# In[69]:


# the venues.
results = requests.get(url).json()
results


# In[72]:


venues = results['response']['groups'][0]['items']
venues


# In[73]:


# Normalize the JSON response
city_venues = json_normalize(venues)
city_venues


# In[74]:


# Filter out the venue name, category, latitude and logitude.
venue_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
city_venues = city_venues.loc[:, venue_columns]
city_venues


# In[75]:


# Change the column names after '.'.
city_venues.columns = [column.split(".")[-1] for column in city_venues.columns]
city_venues


# In[76]:


# Extract the category names from a row.
# We will get the first item in the categories list and then get its name.

def get_category(row):
    categories_list = row['categories']
    if categories_list:
        return categories_list[0]['name']
    return None


# In[77]:


# Replace the values in categories column with the first catogory name.
city_venues['categories'] = city_venues.apply(get_category, axis=1)
city_venues


# In[80]:


venues_list = list()

for name, lat, lng in zip(data['ciudad'], data['X'], data['Y']):
    print("Pizza place:", name)
    
    # Create API request URL
    limit = 100
    radius = 500
    explore_url_prefix = 'https://api.foursquare.com/v2/venues/explore'
    url = '{}?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        explore_url_prefix, CLIENT_ID, CLIENT_SECRET, VERSION, 
        lat, lng, radius, limit)
    
    # Make the request
    city_venues = requests.get(url).json()["response"]['groups'][0]['items']
    
    # Add relevant info to venues_list
    venues_list.extend([(
        name, lat, lng,
        v['venue']['name'], 
        v['venue']['location']['lat'], 
        v['venue']['location']['lng'],  
        v['venue']['categories'][0]['name']) for v in city_venues])

print("Terminado")


# In[81]:


len(venues_list)


# In[82]:


milan_venues = pd.DataFrame(venues_list)
milan_venues.columns = [
    'City', 'City Latitude', 'City Longitude', 
    'Venue', 'Venue Latitude', 'Venue Longitude', 'Venue Category']


# In[84]:


print(milan_venues.shape)
milan_venues.head()


# In[86]:


milan_venues.groupby('City').count()


# In[87]:


print('There are {} uniques categories'.format(
    len(milan_venues['Venue Category'].unique())))


# In[92]:


milan_onehot = pd.get_dummies(milan_venues[['Venue Category']], prefix="", prefix_sep="")
milan_onehot.head(100)


# In[93]:


# Add neighborhood column back to dataframe as NeighborhoodName
milan_onehot.insert(0, 'CityName', milan_venues['City']) 
milan_onehot.head()


# In[95]:


milan_onehot.shape


# In[96]:


venues_count, categories_count = milan_onehot.shape
print("So we have {} different cities and {} categories.".format(
    venues_count, categories_count))


# In[97]:


milan_grouped = milan_onehot.groupby('CityName').mean().reset_index()
milan_grouped


# In[102]:


num_top_venues = 5
for city in milan_grouped['CityName']:
    print("------{}------".format(city))
    temp = milan_grouped[milan_grouped['CityName'] == city].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[111]:


def return_most_common_cities(row, num_top_cities):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_cities]


# In[126]:


num_top_cities=10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['City']
for ind in np.arange(num_top_cities):
    try:
        columns.append('{}{} Most Common Commerce in City'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Commerce in City'.format(ind+1))

# create a new dataframe
city_venues_sorted = pd.DataFrame(columns=columns)
city_venues_sorted['City'] = milan_grouped['CityName']

for ind in np.arange(milan_grouped.shape[0]):
    city_venues_sorted.iloc[ind, 1:] = return_most_common_cities(
        milan_grouped.iloc[ind, :], num_top_cities)

city_venues_sorted


# # Here we have the most common commerce en every city in Milan  in this case we see that is more common in Italy the Pizza place

# In[127]:


# Set number of clusters
k = 5

# Drop the neighborhood name column so that each column contains only the feature set.
milan_grouped_clustering = milan_grouped.drop('CityName', 1)

# Run k-means clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(milan_grouped_clustering)

# Check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[128]:



# Add clustering labels
city_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[138]:


milan_merged = milan_data

# Merge milan_grouped with milan_data to add latitude/longitude for each neighborhood
milan_merged = milan_merged.join(city_venues_sorted.set_index('City'), on='ciudad')

print(milan_merged.shape)
milan_merged.head()


# In[142]:


# Create a map instance
map_milan = folium.Map(
    location=[location.latitude, location.longitude],
    zoom_start=11
,
   tiles='CartoDB dark_matter')

# Set color scheme for the clusters
x = np.arange(k)
ys = [i + x + (i*x)**2 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# Add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
        milan_merged['X'], milan_merged['Y'],
        milan_merged['ciudad'], milan_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_milan)
       
map_milan


# # #Here we have the map with the location of every venue or city in Milan and give us the commerce more common in the area , next we have the segment of the areas that i divided for distance or is more close at the comerce more common.
# 
# ## In this case i used the clusterin with k-mean to giveme the probability of the commerce more common for every area

# In[146]:


cluster_0 = milan_merged.loc[milan_merged['Cluster Labels'] == 0,
                               milan_merged.columns[
                                   [1] + list(range(
                                       5, milan_merged.shape[1]))]]
cluster_0


# In[147]:


cluster_1 = milan_merged.loc[milan_merged['Cluster Labels'] == 1,
                               milan_merged.columns[
                                   [1] + list(range(
                                       5, milan_merged.shape[1]))]]
cluster_1


# In[148]:


cluster_2 = milan_merged.loc[milan_merged['Cluster Labels'] == 2,
                               milan_merged.columns[
                                   [1] + list(range(
                                       5, milan_merged.shape[1]))]]
cluster_2


# In[149]:


cluster_3 = milan_merged.loc[milan_merged['Cluster Labels'] == 3,
                               milan_merged.columns[
                                   [1] + list(range(
                                       5, milan_merged.shape[1]))]]
cluster_3


# In[150]:


cluster_4 = milan_merged.loc[milan_merged['Cluster Labels'] == 4,
                               milan_merged.columns[
                                   [1] + list(range(
                                       5, milan_merged.shape[1]))]]
cluster_4


# In[ ]:




