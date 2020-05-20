#!/usr/bin/env python
# coding: utf-8

# We have built a dataframe of the postal code of each neighborhood along with the borough name and neighborhood name, in order to utilize the Foursquare location data, we need to get the latitude and the longitude coordinates of each neighborhood.

# In[3]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
List_url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
source = requests.get(List_url).text
soup = BeautifulSoup(source, 'xml')
table=soup.find('table')
column_names=['Postalcode','Borough','Neighbourhood']
df = pd.DataFrame(columns=column_names)
for tr_cell in table.find_all('tr'):
    row_data=[]
    for td_cell in tr_cell.find_all('td'):
        row_data.append(td_cell.text.strip())
    if len(row_data)==3:
        df.loc[len(df)] = row_data

        
       


# In[4]:


df.head()


# In[5]:


df=df[df['Borough']!='Not assigned']


# In[6]:


df['Borough']=df[df['Neighbourhood']=='Not assigned']
df.head()


# In[7]:


temp_df=df.groupby('Postalcode')['Neighbourhood'].apply(lambda x: "%s" % ', '.join(x))
temp_df=temp_df.reset_index(drop=False)
temp_df.rename(columns={'Neighbourhood':'Neighbourhood_joined'},inplace=True)


# In[8]:


df_merge = pd.merge(df, temp_df, on='Postalcode')


# In[9]:


df_merge.drop(['Neighbourhood'],axis=1,inplace=True)
df_merge.drop_duplicates(inplace=True)
df_merge.rename(columns={'Neighbourhood_joined':'Neighbourhood'},inplace=True)
df_merge.head()


# In[10]:


df_merge.shape


# In[11]:


def get_geocode(postal_code):
    # initialize your variable to None
    lat_lng_coords = None
    while(lat_lng_coords is None):
        g = geocoder.google('{}, Toronto, Ontario'.format(postal_code))
        lat_lng_coords = g.latlng
    latitude = lat_lng_coords[0]
    longitude = lat_lng_coords[1]
    return latitude,longitude


# In[12]:


geo_df=pd.read_csv('http://cocl.us/Geospatial_data')


# In[13]:


geo_df.head()


# In[14]:


geo_df.rename(columns={'Postal Code':'Postalcode'},inplace=True)
geo_merged = pd.merge(geo_df, df_merge, on='Postalcode')


# In[15]:


geo_data=geo_merged[['Postalcode','Borough','Neighbourhood','Latitude','Longitude']]


# In[16]:


geo_data.head(12)


# In[ ]:




