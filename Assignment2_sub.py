
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians


# In[2]:


# First lets read the data into table format so that we can draw some insight
# Next we will clean the data based on some observations.
data = pd.read_csv('train.csv', nrows=10000000)


# In[3]:


# Lets see what kind of data we have
data.describe()


# In[4]:


data[(data['passenger_count'] > 8) | (data['passenger_count'] <= 0)].shape


# In[5]:


# First lets check the variation of latitude and longitude by drawing the scatter plot taking 
# very relaxed boundary conditions
lat_border = (39, 42.5)
long_border = (-72.03, -75.75)
data.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', 
       color='blue', s=.02, alpha=.6)
plt.title("Pickups")
plt.ylim(lat_border)
plt.xlim(long_border)


# In[6]:


lat_border = (39, 42.5)
long_border = (-72.03, -75.75)
data.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', 
       color='green', s=.02, alpha=.6)
plt.title("Dropoffs")
plt.ylim(lat_border)
plt.xlim(long_border)


# In[7]:


#Plotting histograms as well to take a more closer look
data[(data.pickup_latitude > 40) & (data.pickup_latitude < 42)].pickup_latitude.hist(bins=50, figsize=(11,8))


# In[8]:


data[(data.dropoff_latitude > 40) & (data.dropoff_latitude < 42)].dropoff_latitude.hist(bins=50, figsize=(11,8))


# In[9]:


data[(data.pickup_longitude > -74.5) & (data.pickup_longitude < -72)].pickup_longitude.hist(bins=50, figsize=(11,8))


# In[10]:


data[(data.dropoff_longitude > -74.5) & (data.dropoff_longitude < -72)].dropoff_longitude.hist(bins=50, figsize=(11,8))


# In[11]:


data['passenger_count'].hist(bins=64, figsize=(16,8))


# In[12]:


# This method will be used to do the initial phase opf the data cleaning, 
# primarily outlier removals and some approximations.
def clean_data(df):
    # New york city has a central location cordinates of Latitude = 40.7128 and Longitude = 74.0060
    # Therefore we restrict the pickup and dropoff locations to avoid considering rides which
    # are lying outside the nyc area.
    nyc_min_latitude = 40.45
    nyc_max_latitude = 40.97
    nyc_min_longitude = -74.28
    nyc_max_longitude = -73.64
    
    # Removing null entries from the data
    df = df.dropna(how='any', axis = 'rows')
    
    # Removing entries for which pickup/ dropoff locations do not lie inside the nyc area
    df = df[(df['pickup_latitude'] >= nyc_min_latitude) & (df['pickup_latitude'] <= nyc_max_latitude)]
    df = df[(df['pickup_longitude'] >= nyc_min_longitude) & (df['pickup_longitude'] <= nyc_max_longitude)]
    df = df[(df['dropoff_latitude'] >= nyc_min_latitude) & (df['dropoff_latitude'] <= nyc_max_latitude)]
    df = df[(df['dropoff_longitude'] >= nyc_min_longitude) & (df['dropoff_longitude'] <= nyc_max_longitude)]
    
    # Clean entries which have passenger count greater than 8
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 8)]
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]
    return df


# In[13]:


# Lets clean the data now to remove outliers and reduce our data to meaningful entries
data = clean_data(data)


# In[14]:


data.shape


# In[15]:


stat = data.describe()


# In[16]:


stat[['pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude']]


# In[17]:


# There has been some reduction in the dataset. (Irrelevant data)
data.shape


# In[18]:


# This code has been took from stack_overflow and gives the 
# haversine distance between two points on the earth.

def get_euclidean_dist(loc_data):
    orig_lat , orig_lon, dest_lat, dest_lon = loc_data
                           
    radius = 6371    # This is a constant whose value is equal to Earth's radius
                           
    deltaLat = radians(dest_lat-orig_lat)
    deltaLon = radians(dest_lon-orig_lon)
    
    a = sin(deltaLat/2)**2 + cos(radians(orig_lat)) * cos(radians(dest_lat)) * sin(deltaLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = radius * c
    return d


# In[19]:


data.head()


# In[20]:


# Calculate the distance of each ride and make another entry in the dataset. This is an important field
# as fare amount is directly related to the distance travelled
columns = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
data['distance'] = data[columns].apply(get_euclidean_dist, axis=1)


# In[21]:


data.head(5)


# In[22]:


data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# In[23]:


pdt = data['pickup_datetime']


# In[24]:


dtIdx = pd.DatetimeIndex(data['pickup_datetime'])


# In[25]:


## This function will give us the absolute time of the day in minuites. 
## This is done to see if incorporating minutes in the time improves the correlation
def get_time_of_day(dateTime):
    dtIdx = pd.DatetimeIndex(dateTime)
    hours = dtIdx.hour
    minutes = dtIdx.minute
    absTimeofDay = (60*hours) + minutes
    return absTimeofDay


# In[26]:


data['time'] = get_time_of_day(data['pickup_datetime'])


# In[27]:


data.head()


# In[28]:


## Now we will see the relationship between the distance and the fare amount
## Since it is logical that a shorter ride will cost less than a longer one, 
# based on this data we can actually get some insightful information
data.plot(kind='scatter',x='distance',y='fare_amount', s=0.2, alpha=0.4)


# In[29]:


# Preprocessing data to fetch some useful information regarding the time of the ride/ temporal analysis
data['year'] = pd.to_datetime(data['pickup_datetime']).dt.year
data['month'] = pd.to_datetime(data['pickup_datetime']).dt.month
data['day'] = pd.to_datetime(data['pickup_datetime']).dt.day
data['hour'] = pd.to_datetime(data['pickup_datetime']).dt.hour


# In[30]:


# 
data.plot(kind='scatter',x='hour',y='distance', s=0.2, alpha=0.4)
plt.title("Time of the day(min) vs distance")


# In[31]:


data.plot(kind='scatter',x='hour',y='fare_amount', s=0.2, alpha=0.4)
plt.title("Time of the day(min) vs fare amount")


# In[32]:


data[['fare_amount','distance','time', 'hour']].corr()


# In[33]:


data['distance'].corr(data['fare_amount'])


# In[34]:


data.head()


# In[35]:


data['rate'] = data['fare_amount']/data['distance']


# In[36]:


data = data[(data['rate'] > 0.5) & (data['rate'] < 10)]


# In[37]:


data.plot(kind='scatter',x='distance',y='fare_amount', s=0.2, alpha=0.4)


# In[72]:


data[['fare_amount','distance','time', 'hour']].corr()


# In[39]:


data.shape


# In[40]:


# Lets see if there is some relationship b/w the day of the week and number of rides taken.
# Usually more taxi's are booked during weekday.
data['day'].hist(bins=100, figsize=(16,8))
data['weekday'] = pd.to_datetime(data['pickup_datetime']).dt.weekday


# In[41]:


data['weekday'].hist(bins=32, figsize=(16,8))


# In[42]:


# Plotting boxplot to get information regarding the distribution of data . Plotting for
# 1.Finding out if there exist a relation between the hour of taxi pickup and fare amount
# 2.Finding out if there exist a relation between the month of taxi pickup and fare amount
# 3.Finding out if there exist a relation between the year of taxi pickup and fare amount

data[['fare_amount','month']].boxplot(by='month',showfliers=False)
data[['fare_amount','day']].boxplot(by='day',showfliers=False)
data[['distance','hour']].boxplot(by='hour',showfliers=False)
data[['fare_amount','hour']].boxplot(by='hour',showfliers=False)

# From the below plot we can see that the fare is quite high in the morning hours, this could be because of 
# airport rides as the same relation exist for the hour vs distance plot. Lets try to find if these rides are
# actually corresponding to airport pickups and drops.


# In[43]:


# Here we check if the given cordinates of the pickup location or dropoff location is matching
# with the airports near the new york city. There are 3 airports near the city:
# 1. JFK 
# 2. Laguardia 
# 3. Newark
# Getting cordinates of these airports from the web
# JFK ->    Latitude: 40.6413111,   Longitude: -73.7781391
# LaG ->    Latitude: 40.77725,     Longitude: -73.872611
# Newark ->  Latitude: 40.6925,     Longitide: -74.168611


def get_is_airport_ride(column):
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = column
    jfk_airport = (40.6413, -73.778)
    lag_airport = (40.777, -73.872)
    newrk_airport = (40.692, -74.168)
    
    is_drop_at_jkf = (pickup_lat, pickup_lon, jfk_airport[0], jfk_airport[1] )
    is_drop_at_lag = (pickup_lat, pickup_lon, lag_airport[0], lag_airport[1] )
    is_drop_at_newrk = (pickup_lat, pickup_lon, newrk_airport[0], newrk_airport[1] )
    
    is_pickup_from_jkf = (jfk_airport[0], jfk_airport[1],dropoff_lat, dropoff_lon )
    is_pickup_from_lag = (lag_airport[0], lag_airport[1],dropoff_lat, dropoff_lon )
    is_pickup_from_newrk = (newrk_airport[0], newrk_airport[1], dropoff_lat, dropoff_lon)
    
    if(get_euclidean_dist(is_pickup_from_jkf) < 1 or get_euclidean_dist(is_pickup_from_lag) < 1 or 
       get_euclidean_dist(is_pickup_from_newrk) < 1 ):
        return 1
    if(get_euclidean_dist(is_drop_at_jkf) < 1 or get_euclidean_dist(is_drop_at_lag) < 1 or 
       get_euclidean_dist(is_drop_at_newrk) < 1 ):
        return 0
    
    return 0


# In[44]:


# Adding a new feature, which will give us information about airport rides
# Based on this information, I plan to do further analysis on the pattern of long distance rides
columns = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
data['is_airport_ride'] = data[columns].apply(get_is_airport_ride, axis=1)


# In[45]:


data['is_airport_ride'].hist(bins=32, figsize=(16,8))


# In[46]:


data.plot(kind='scatter',x='distance',y='is_airport_ride', s=0.2, alpha=0.4)
plt.title("Distance vs is_airport_ride")


# In[47]:


# Lets find out if there is something interesting we can do with the data we collected from the field is_airport_ride 
# and the hour of the day. This plot will help us analyze the airport taxi traffic throughout a day.
fig, ax = plt.subplots(figsize=(15,7))
data_airport = data[data['is_airport_ride'] > 0]
data_airport.groupby(['hour','is_airport_ride']).count()['key'].unstack().plot(ax=ax)


# In[48]:


features_to_keep = ['pickup_latitude', 'pickup_longitude', 'dropoff_longitude',
       'dropoff_latitude', 'passenger_count', 'distance', 'hour', 'year','is_airport_ride']


# In[49]:


train_data = data[features_to_keep]


# In[50]:


train_data.head()


# In[51]:


output_data = data['fare_amount']


# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(train_data, output_data, test_size = 0.2, random_state=42)


# In[54]:


linear_reg = LinearRegression()


# In[55]:


linear_reg.fit(X_train, y_train)


# In[56]:


print(linear_reg.coef_)


# In[57]:


y_pred = linear_reg.predict(X_test)


# In[58]:


rmse = np.sqrt(mean_squared_error(y_test , y_pred))
print("root mean Squared error: {}".format(rmse))


# In[59]:


test_data = pd.read_csv('test.csv')


# In[60]:


test_data.head()


# In[61]:


test_data.count()


# In[62]:


# just like training data , we add the 'distance' field to the test data as part of preprocessing
columns = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
test_data['distance'] = test_data[columns].apply(get_euclidean_dist, axis=1)
test_data['is_airport_ride'] = test_data[columns].apply(get_is_airport_ride, axis=1)


# In[63]:


# Making test data same as training data in terms of representation so that the model 
# doesn't complain.
test_data['hour'] = test_data.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
test_data['year'] = test_data.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
# Keeping this as it is requiresd to be output in the submission.csv
test_data_with_key = test_data[['key'] + features_to_keep] 
test_data = test_data[features_to_keep]


# In[64]:


test_data.head()


# In[65]:


# Getting the prediction results from the linear regressor model and output it to the submission file
linear_reg.fit(train_data, output_data)
test_predictions = linear_reg.predict(test_data)


# In[66]:


len(test_predictions)


# In[67]:


submission = pd.DataFrame(
    {'key': test_data_with_key.key, 'fare_amount': test_predictions},
    columns = ['key', 'fare_amount'])


# In[68]:


submission.to_csv('submission.csv', index = False)


# In[69]:


# Got a score of 5.35 with k-fold Linear regression 
# Now trying random Forest regressor to check if there is any improvement
from sklearn.ensemble import RandomForestRegressor
rfgModel = RandomForestRegressor()
# Trying cross validation first to check if the model is givibng good results. Root mean square value is
# a good approximation of the performance of a prediction model
print("Random Forest Generator Parameters: ")
print(rfgModel.get_params() )
rfgModel.fit(X_train, y_train)
rfgModel_pred = rfgModel.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test , rfgModel_pred))
print("root mean Squared error: {}".format(rmse))
rfgModel.fit(train_data, output_data)
# Now running the model on actual data test data
rfgModel_pred = rfgModel.predict(test_data)


# In[70]:


submission = pd.DataFrame(
    {'key': test_data_with_key.key, 'fare_amount': rfgModel_pred},
    columns = ['key', 'fare_amount'])


# In[71]:


submission.to_csv('submission.csv', index = False)

