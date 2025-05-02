# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Example dataset (you can replace this with real tweet data)
data = {
    'text': [
        "I love this product! It’s amazing.",
        "This is the worst service I’ve ever experienced.",
        "Not bad, but could be better.",
        "Absolutely fantastic! Will buy again.",
        "Terrible packaging. Very disappointed.",
        "Pretty good for the price.",
        "I have mixed feelings about this.",
        "Totally worth it! Great quality.",
        "Waste of money. Do not recommend.",
        "I’m so happy with the results!"
    ]
}

df = pd.DataFrame(data)

# ------------------------
# 1. Sentiment Analysis
# ------------------------
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['sentiment'] = df['text'].apply(get_sentiment)

# Categorize sentiment
def label_sentiment(score):
    if score > 0.2:
        return 'Positive'
    elif score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['sentiment'].apply(label_sentiment)

# ------------------------
# 2. Visualization
# ------------------------

# Sentiment count
sns.countplot(x='sentiment_label', data=df, palette='pastel')
plt.title('Sentiment Classification')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

# Sentiment score distribution
sns.histplot(df['sentiment'], kde=True, bins=10, color='skyblue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Display sample sentiment-tagged tweets
print("\nSample Sentiment Labels:\n", df[['text', 'sentiment', 'sentiment_label']])