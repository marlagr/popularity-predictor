import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import requests
import matplotlib.pyplot as plt
import plotly.express as px 
import statsmodels.api as sm
from scipy.stats import sem

# Dataset variables
features=["danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence"]
dataset = pd.read_csv('./archive/tracks.csv', usecols=["popularity","danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence"])

dataset.replace([',,', 0], np.nan, inplace=True)
dataset.replace(['', 0], np.nan, inplace=True)
dataset.dropna(inplace=True)
#dataset = dataset.drop(dataset.index[range(dif)])
print(dataset.head())
# Dependent Variables 
df_x= dataset.loc[:,dataset.columns!='popularity']
# Independent Variables 
df_y = dataset["popularity"]

# size for the test
sizeTest = math.floor(df_y.size * 0.25)

# Split the data into training/testing sets
#X
x_train = df_x[:sizeTest]
x_test = df_x[sizeTest:]
#Y
y_train = df_y[:sizeTest]
y_test = df_y[sizeTest:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model
model = regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

print("Results")
print('Coefficients: \n', regr.coef_)
print('Mean squared error: \n %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: \n %.2f' % r2_score(y_test, y_pred))

# Check values
print("10 First Predicted Vs Expected")
for i in range(10):
  print("Predicted: ", y_pred[i], " Expected: ", df_y.values[i])
print("10 Last values Predicted Vs Expected")
for i in range(len(y_pred)-10, len(y_pred), 1):
  print("Predicted: ", y_pred[i], " Expected: ", df_y.values[i])

print("\nPredict the popularity of a song, get the token from the Spotify Console")

prediction = 1
access_token = input("Write your access token: ")

while prediction != 0:
    # Track ID from the URI
    track_id = input("Write the track ID from the URL: ")

    BASE_URL = 'https://api.spotify.com/v1/'

    # actual GET request with proper header
    headers = {
        'Authorization': 'Bearer {token}'.format(token=access_token)
    }

    r = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)

    r = r.json()

    print("Track atributes\n")
    print(r)

    # Create data frame with predictions
    df_pred = pd.DataFrame([(r["danceability"],r["energy"],r["loudness"],r["speechiness"],r["acousticness"],r["instrumentalness"],r["liveness"],r["valence"])], columns = features)
    print("\nPopularity Probability: \n")
    print(regr.predict(df_pred)[0]) 

    print("\nAnother Prediction?")
    prediction = int(input("1 for yes, 0 for no\n"))