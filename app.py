from flask import Flask, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)

# Load series data from CSV
series_df = pd.read_csv('static/app.csv')

series_df.dropna(inplace=True)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Extract features from genres, titles, actors, and directors
features = series_df['Genre']  + ' ' + series_df['Actors'] + ' ' + series_df['Director']+series_df['Tags']
feature_matrix = vectorizer.fit_transform(features)

@app.route('/',methods=["POST"])
def recommend():
    data =requests.json()
    names = data.get('name',[])
    genres = data.get('genres',[])
    tags = data.get('tags',[])





    # User preferences (example)
    user_preferences = {
        'Genres': ' '.join(genres),
        'Actors': ['Joaquin Phoenix', 'Robert De Niro', 'Zazie Beetz', 'Frances Conroy']
,
        'Director': ['José Esteban Alenda', 'César Esteban Alenda'],
        'Tags': ' '.join(tags),

    }

    # Create user profile
    user_profile = ' '.join(user_preferences['Genres']) + ' ' + ' '.join(user_preferences['Actors']) + ' ' + ' '.join(user_preferences['Director'])+ ' ' + ' '.join(user_preferences['Director'])
    user_profile_matrix = vectorizer.transform([user_profile])

    # Calculate similarity scores
    similarity_scores = cosine_similarity(user_profile_matrix, feature_matrix)

    # Get recommendations 
    recommendations = []
    for i, score in enumerate(similarity_scores[0]):
           series_title = series_df.iloc[i]['Title']
           series_img = series_df.iloc[i]['Images']
           recommendations.append({'Title': series_title, 'Similarity Score': score,'Images':series_img})

    # Sort recommendations by similarity score
    recommendations = sorted(recommendations, key=lambda x: x['Similarity Score'], reverse=True)

    # Return recommendations as JSON
    return jsonify(recommendations[:10])  # Return top 3 recommendations

if __name__ == '__main__':  
    app.run(debug=True)
