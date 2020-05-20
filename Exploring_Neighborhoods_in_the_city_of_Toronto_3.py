#!/usr/bin/env python
# coding: utf-8

# Explore and cluster the neighborhoods in Toronto.
# Generate maps to visualize your neighborhoods and how they cluster togethe

# In[78]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install beautifulsoup4')
get_ipython().system('{sys.executable} -m pip install requests')
get_ipython().system('{sys.executable} -m pip install geopy')
get_ipython().system('{sys.executable} -m pip install sklearn')
get_ipython().system('{sys.executable} -m pip install folium')
get_ipython().system('{sys.executable} -m pip install matplotlib')


# In[80]:


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


# In[61]:


# Get the HTML text from the wiki page
wiki_url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
html_text = requests.get(wiki_url).text

# Extract the table data using BeautifulSoup
soup = BeautifulSoup(html_text)
table = soup.find('table', attrs={'class':'wikitable sortable'})
trs = table.find_all('tr')

# Extract the text from all the table cells and add all rows
# to a list of rows.
rows = list()
for tr in trs:
    td = tr.find_all('td')
    row = [ele.text.strip() for ele in td]
    if row:
        # Ignore empty rows with no 'td',
        # applicable for the column headers row.
        rows.append(row)
        
# Convert the data to a pandas dataframe with 
# columns 'PostalCode', 'Borough', and 'Neighborhood'.
df = pd.DataFrame(rows,
                  columns=['PostalCode', 'Borough', 'Neighborhood'])

# Remove rows with column 'Not assigned' for column 'Borough'
df = df[df.Borough != 'Not assigned']
df.reset_index(inplace=True, drop=True)

# If a cell has a borough but a Not assigned neighborhood, 
# then the neighborhood will be the same as the borough.
# So, replace 'Neighborhood' columns with value as
# 'Not assigned' with the value of its 'Borough'
df['Neighborhood'] = df.apply(
    lambda row: 
    row['Borough'] if row['Neighborhood'] == 'Not assigned' 
    else row['Neighborhood'],
    axis=1)

# More than one neighborhood can exist in one postal code area.
# Combine rows with the same postal code into a single row.
# For example, in the table on the Wikipedia page, M5A is 
# listed twice and has two neighborhoods: Harbourfront and
# Regent Park. These two rows will be combined into one row
# with the neighborhoods separated with a comma.
df = df.groupby(['PostalCode', 'Borough'])['Neighborhood'].    apply(', '.join).to_frame()
df.reset_index(inplace=True)


# In[4]:


soup = BeautifulSoup(source, 'xml')
table=soup.find('table')
column_names=['Postalcode','Borough','Neighborhood']
df = pd.DataFrame(columns=column_names)


# In[62]:


df.head(12)


# In[5]:


for tr_cell in table.find_all('tr'):
    row_data=[]
    for td_cell in tr_cell.find_all('td'):
        row_data.append(td_cell.text.strip())
    if len(row_data)==3:
        df.loc[len(df)] = row_data
    


# In[63]:


print("Dataframe shape: ", df.shape)


# In[71]:


#Read the CSV data to a pandas dataframe.
geospatial_data = pd.read_csv('https://cocl.us/Geospatial_data')

## Combine the two dataframes.
# Concatenate the two dataframes using column "PostalCode" from 'df' and
# "Postal Code" from 'geospatial_data'.
df = pd.concat(
   [df.set_index('PostalCode'), geospatial_data.set_index('Postal Code')],
   axis=1, join='inner')


# Postal code will now be the index, change it to a non-index 
# column and reset index.
df.reset_index(inplace=True)

# Rename the 'index' column to 'PostalCode'
df.rename(columns={'index': 'PostalCode'}, inplace=True)


# In[69]:


df.head(12)


# In[72]:


print("Dataframe shape: ", df.shape)


# In[73]:


address = 'Toronto, Canada'
geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
print('The geograpical coordinate of Toronto are {}, {}.'.format(
    location.latitude, location.longitude))


# In[81]:


map_toronto = folium.Map(
    location=[location.latitude, location.longitude],
    zoom_start=10)
map_toronto


# In[82]:


for _, row in df.iterrows():
    label = '{} ({}), {}'.format(
        row.PostalCode, row.Neighborhood, row.Borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [row.Latitude, row.Longitude],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto) 
    
map_toronto


# In[83]:


# Create dataframe with boroughs containing the term 'Toronto'
toronto_data = df[df.Borough.str.contains('Toronto')]
toronto_data.reset_index(inplace=True, drop=True)
toronto_data


# In[84]:


toronto_data.shape


# In[85]:


# Create a map instance
map_toronto = folium.Map(
    location=[location.latitude, location.longitude],
    zoom_start=11)

# Add markers for all 38 neighborhoods
for _, row in toronto_data.iterrows():
    label = '{} ({}), {}'.format(
        row.PostalCode, row.Neighborhood, row.Borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [row.Latitude, row.Longitude],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto) 
    
map_toronto


# In[87]:


CLIENT_ID = 'RR3ZMGRDZADKUZOODJ3KAUXVH0KOLPSQ4JTBSAQPC2Z5PTIU'
CLIENT_SECRET ='PVE2JCJVCSCFPWUB0S25SQQH5GBDSTKYM34G1QZZC5UVNHGM'
VERSION = '20190601'


# In[88]:


neighborhood_name = toronto_data.loc[0, 'Neighborhood']
neighborhood_latitude = toronto_data.loc[0, 'Latitude']
neighborhood_longitude = toronto_data.loc[0, 'Longitude']
print("Latitude and longitude of neighborhood '{}' are [{}, {}]".format(
    neighborhood_name, neighborhood_latitude, neighborhood_longitude))


# In[90]:


# Construct the URL
limit = 100
radius = 500
explore_url_prefix = 'https://api.foursquare.com/v2/venues/explore'
url = '{}?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    explore_url_prefix, CLIENT_ID, CLIENT_SECRET, VERSION, 
    neighborhood_latitude, neighborhood_longitude, radius, limit)


# In[91]:


# the venues.
results = requests.get(url).json()
results


# In[92]:


venues = results['response']['groups'][0]['items']
venues


# In[93]:


# Normalize the JSON response
neighborhood_venues = json_normalize(venues)
neighborhood_venues


# In[94]:


# Filter out the venue name, category, latitude and logitude.
venue_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
neighborhood_venues = neighborhood_venues.loc[:, venue_columns]
neighborhood_venues


# In[95]:


# Change the column names after '.'.
neighborhood_venues.columns = [column.split(".")[-1] for column in neighborhood_venues.columns]
neighborhood_venues


# In[96]:


# Extract the category names from a row.
# We will get the first item in the categories list and then get its name.

def get_category(row):
    categories_list = row['categories']
    if categories_list:
        return categories_list[0]['name']
    return None


# In[97]:


# Replace the values in categories column with the first catogory name.
neighborhood_venues['categories'] = neighborhood_venues.apply(get_category, axis=1)
neighborhood_venues


# In[98]:


venues_list = list()

for name, lat, lng in zip(toronto_data['Neighborhood'], toronto_data['Latitude'], toronto_data['Longitude']):
    print("Collecting venues for neighborhood:", name)
    
    # Create API request URL
    limit = 100
    radius = 500
    explore_url_prefix = 'https://api.foursquare.com/v2/venues/explore'
    url = '{}?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        explore_url_prefix, CLIENT_ID, CLIENT_SECRET, VERSION, 
        lat, lng, radius, limit)
    
    # Make the request
    neighborhood_venues = requests.get(url).json()["response"]['groups'][0]['items']
    
    # Add relevant info to venues_list
    venues_list.extend([(
        name, lat, lng,
        v['venue']['name'], 
        v['venue']['location']['lat'], 
        v['venue']['location']['lng'],  
        v['venue']['categories'][0]['name']) for v in neighborhood_venues])

print("Terminado")


# In[99]:


len(venues_list)


# In[100]:


toronto_venues = pd.DataFrame(venues_list)
toronto_venues.columns = [
    'Neighborhood', 'Neighborhood Latitude', 'Neighborhood Longitude', 
    'Venue', 'Venue Latitude', 'Venue Longitude', 'Venue Category']


# In[101]:


print(toronto_venues.shape)
toronto_venues.head()


# In[102]:


toronto_venues.groupby('Neighborhood').count()


# In[103]:


print('There are {} uniques categories'.format(
    len(toronto_venues['Venue Category'].unique())))


# In[104]:


toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
toronto_onehot.head()


# In[105]:


# Add neighborhood column back to dataframe as NeighborhoodName
toronto_onehot.insert(0, 'NeighborhoodName', toronto_venues['Neighborhood']) 
toronto_onehot.head()


# In[106]:


toronto_onehot.shape


# In[107]:


venues_count, categories_count = toronto_onehot.shape
print("So we have {} different venues and {} venue categories.".format(
    venues_count, categories_count))


# In[108]:


toronto_grouped = toronto_onehot.groupby('NeighborhoodName').mean().reset_index()
toronto_grouped


# In[109]:


num_top_venues = 5
for neighborhood in toronto_grouped['NeighborhoodName']:
    print("------{}------".format(neighborhood))
    temp = toronto_grouped[toronto_grouped['NeighborhoodName'] == neighborhood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[110]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[111]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['NeighborhoodName']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(
        toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# In[112]:


# Set number of clusters
k = 5

# Drop the neighborhood name column so that each column contains only the feature set.
toronto_grouped_clustering = toronto_grouped.drop('NeighborhoodName', 1)

# Run k-means clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(toronto_grouped_clustering)

# Check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[113]:


# Add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = toronto_data

# Merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

print(toronto_merged.shape)
toronto_merged.head()


# In[114]:


# Create a map instance
map_toronto = folium.Map(
    location=[location.latitude, location.longitude],
    zoom_start=11)

# Set color scheme for the clusters
x = np.arange(k)
ys = [i + x + (i*x)**2 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# Add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
        toronto_merged['Latitude'], toronto_merged['Longitude'],
        toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_toronto)
       
map_toronto


# In[115]:


cluster_0 = toronto_merged.loc[toronto_merged['Cluster Labels'] == 0,
                               toronto_merged.columns[
                                   [2] + list(range(
                                       5, toronto_merged.shape[1]))]]
cluster_0


# In[116]:


print(cluster_0.shape)


# In[117]:


#cluster 1
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1,
                   toronto_merged.columns[
                       [2] + list(range(5, toronto_merged.shape[1]))]]


# In[118]:


#cluster 2
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2,
                   toronto_merged.columns[
                       [2] + list(range(5, toronto_merged.shape[1]))]]


# In[119]:


#cluster 3
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3,
                   toronto_merged.columns[
                       [2] + list(range(5, toronto_merged.shape[1]))]]


# In[120]:


#cluster 4
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4,
                   toronto_merged.columns[
                       [2] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:




