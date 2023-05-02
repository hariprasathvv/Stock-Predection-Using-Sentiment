import os
import pandas as pd
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Set path to folder containing pre-processed tweet files
folder_path = 'C:/Users/Hari/Desktop/ML/AAPL'

# Get list of file names in the folder
file_names = os.listdir(folder_path)

# Create an empty list to store sentiment scores for each date
sentiment_scores = []

# Loop through each file in the folder
for file_name in file_names:
    # Get the date from the file name (assuming file name format is 'YYYY-MM-DD.txt')
    date = file_name.split('.')[0]
    
    try:
        # Read the file line by line and extract tweet text from each JSON object
        with open(os.path.join(folder_path, file_name), 'r', encoding='ascii') as f:
            tweets = []
            for line in f:
                tweet_obj = json.loads(line)
                tweet_text = ' '.join(tweet_obj['text'])
                tweets.append(tweet_text)
        
        # Apply VADER sentiment analysis to each tweet in the file
        scores = [analyzer.polarity_scores(tweet) for tweet in tweets]
        
        # Compute the mean sentiment score for the file
        mean_score = sum([score['compound'] for score in scores]) / len(scores)
        
        
        # Append the mean score and date to the sentiment_scores list
        sentiment_scores.append({'date': date, 'score': mean_score})
    
    except Exception as e:
        print(f'Error processing file {file_name}: {e}')

# Convert sentiment_scores list to a pandas DataFrame
sentiment_df = pd.DataFrame(sentiment_scores)

# Format date column as DD-MM-YYYY and write DataFrame to CSV file
sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.strftime('%d-%m-%Y')
sentiment_df.to_csv('sentiment_scores.csv', index=False)

# Print the DataFrame
print(sentiment_df)
