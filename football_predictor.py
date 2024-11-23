# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st

# 1. Data Collection from Football API
def fetch_data(api_url, api_key):
    headers = {"X-Auth-Token": api_key}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data['matches']
    else:
        print("Error fetching data:", response.status_code)
        return None

# Example API URL and API key (replace with your own)
API_URL = "https://api.football-data.org/v2/matches"
API_KEY = "31ed67e0565641f0a395818827ba3129"  # Replace with your API key

# Fetching the data
matches = fetch_data(API_URL, API_KEY)

# 2. Data Preprocessing
df = pd.DataFrame(matches)
df['date'] = pd.to_datetime(df['utcDate'])
df['home_team'] = df['homeTeam.name']
df['away_team'] = df['awayTeam.name']
df['outcome'] = df['score.winner']
df = df[['date', 'home_team', 'away_team', 'outcome']]
df.dropna(inplace=True)

# Feature Engineering
def recent_form(team, df):
    results = df[df['home_team'] == team].tail(5)['outcome']
    return results.map({'HOME_TEAM': 1, 'AWAY_TEAM': 0, 'DRAW': 2}).mean()

df['home_form'] = df['home_team'].apply(lambda x: recent_form(x, df))
df['away_form'] = df['away_team'].apply(lambda x: recent_form(x, df))

# Encoding the outcome as numeric values (home=0, away=1, draw=2)
df['outcome'] = df['outcome'].map({'HOME_TEAM': 0, 'AWAY_TEAM': 1, 'DRAW': 2})

# 3. Sentiment Analysis for News (Optional)
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(news):
    return sia.polarity_scores(news)['compound']

# Example usage of sentiment analysis (you can gather news for teams here)
news_article = "Star player injured ahead of the match."
sentiment_score = analyze_sentiment(news_article)
print(f"Sentiment score: {sentiment_score}")

# 4. Train the Model
X = df[['home_form', 'away_form']]
y = df['outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# 5. Prediction Function
def predict_match(home_team, away_team, model, df):
    home_form = recent_form(home_team, df)
    away_form = recent_form(away_team, df)
    features = [[home_form, away_form]]
    prediction = model.predict(features)
    return prediction

# 6. Streamlit Interface for User Input
st.title("Football Prediction AI")

home_team = st.text_input("Enter Home Team:")
away_team = st.text_input("Enter Away Team:")

if st.button("Predict"):
    if home_team and away_team:
        prediction = predict_match(home_team, away_team, model, df)
        outcome = "Home Win" if prediction == 0 else "Away Win" if prediction == 1 else "Draw"
        st.write(f"Prediction: {outcome}")
    else:
        st.write("Please enter both teams.")
