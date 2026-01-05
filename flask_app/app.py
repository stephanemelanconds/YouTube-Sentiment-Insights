import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
from dotenv import load_dotenv

import os
load_dotenv()  # Load environment variables from .env file

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION')

from googleapiclient.discovery import build

# Load the API_KEY from the .env file
API_KEY = os.getenv('API_KEY')
youtube = build('youtube', 'v3', developerKey=API_KEY)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment



# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-3-238-124-243.compute-1.amazonaws.com:5000/")  # Replace with your MLflow tracking URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
   
    return model, vectorizer

def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-3-238-124-243.compute-1.amazonaws.com:5000")
    
    # Try loading by version
    model_uri = f"models:/{model_name}/{model_version}"
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Registry load failed, attempting direct run load: {e}")
        # FALLBACK: Use the Run ID directly if the registry path is still bugged
        # Replace this ID with the one from your logs: 28155f30930c48b8bbea615db730bee1
        model_uri = "runs:/28155f30930c48b8bbea615db730bee1/lgbm_model"
        model = mlflow.pyfunc.load_model(model_uri)

    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
   
    return model, vectorizer

#def load_model(model_path, vectorizer_path):
#    """Load the trained model."""
#    try:
#        with open(model_path, 'rb') as file:
#            model = pickle.load(file)
#        
#        with open(vectorizer_path, 'rb') as file:
#            vectorizer = pickle.load(file)
#      
#        return model, vectorizer
#    except Exception as e:
#        raise


@app.route('/get_config', methods=['GET'])
def get_config():
    """Sends the API key from the server's .env to the JavaScript frontend."""
    return jsonify({
        "API_KEY": os.getenv('API_KEY')
    })

# Initialize the model and vectorizer
# model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")  

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed

# model, vectorizer = load_model_and_vectorizer("28155f30930c48b8bbea615db730bee1", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to our flask api"



@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # 1. Extract data from JSON
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # 2. Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # 3. Vectorize (Sparse Matrix)
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # 4. FIX: Convert to DataFrame with Feature Names (Schema Enforcement)
        # This is what prevents the 'Failed to enforce schema' error
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
        
        # 5. Predict using the DataFrame
        predictions = model.predict(df_input).tolist() 
        
        # Convert predictions to strings for consistency if your frontend expects strings
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        # Added traceback print to help you debug in the terminal if something else fails
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # 6. Return response
    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp} 
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # 1. Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # 2. Vectorize (this returns a sparse matrix)
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # 3. Create a DataFrame with the correct column names (The Fix)
        # MLflow requires the feature names to match the schema it recorded
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
        
        # 4. Make predictions using the DataFrame
        predictions = model.predict(df_input).tolist() 
        
    except Exception as e:
        # This will now give you more detail if it fails
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response = [{"comment": comment, "sentiment": int(sentiment)} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    


