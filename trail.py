from transformers import pipeline

# Load Hugging Face sentiment analysis model (RoBERTa trained on tweets)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

def analyze_sentiment(comment):
    if not comment.strip():
        return "Neutral"
    result = sentiment_pipeline(comment)[0]['label']
    if result == "LABEL_0":
        return "Negative"
    elif result == "LABEL_1":
        return "Neutral"
    elif result == "LABEL_2":
        return "Positive"
    else:
        return "Neutral"
