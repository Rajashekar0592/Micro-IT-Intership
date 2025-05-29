import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Predefined stopwords list
stopwords = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}

# Balanced dataset with 15 examples per class
data = {
    'text': [
        'I love this product, it is amazing!',
        'Fantastic experience, highly recommend!',
        'Absolutely wonderful product!',
        'Really happy with my purchase!',
        'This is the best thing I’ve ever bought!',
        'Superb quality, I’m thrilled!',
        'Incredible product, exceeded expectations!',
        'Really impressed with this item!',
        'Amazing value for the price!',
        'Best purchase ever, so satisfied!',
        'Great product, works perfectly!',
        'Loving this item, so happy!',
        'Fantastic quality, worth every penny!',
        'Really enjoyable experience!',
        'Top-notch product, very pleased!',
        'This is the worst service ever.',
        'Terrible quality, very disappointed.',
        'Horrible customer support, never again.',
        'Not satisfied, poor performance.',
        'Completely useless, waste of money.',
        'Worst product I’ve ever used!',
        'Really poor quality, avoid this.',
        'Service was a total letdown.',
        'Disappointing purchase, broke quickly.',
        'Pathetic service, totally unacceptable.',
        'Awful product, regret buying it.',
        'Terrible experience, so frustrating.',
        'Dreadful quality, complete failure.',
        'Hated this product, total waste.',
        'Really bad service, not recommended.',
        'It’s an average item, nothing special.',
        'Works fine, no strong opinion.',
        'Decent quality, not impressive.',
        'Service is standard, no complaints.',
        'Just an okay product, does the job.',
        'Nothing unique, fairly typical.',
        'Meets basic expectations, that’s it.',
        'Not bad, not great, just average.',
        'Functionality is adequate, nothing more.',
        'Plain and simple, no excitement.',
        'It’s a typical product, nothing stands out.',
        'Service is okay, nothing remarkable.',
        'Just a regular item, no issues.',
        'Nothing to rave about, just okay.',
        'Standard performance, no surprises.'
    ],
    'sentiment': [
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords]
    # Remove extra whitespace and join
    text = ' '.join(words)
    return text

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer(max_features=150)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model with balanced class weights
model = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.5, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Function to predict sentiment for new text
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(text_vector)[0]
    return prediction

# Example predictions
sample_texts = [
    'This is a great product!',
    'I am not happy with this service.',
    'It is an average item.'
]
predictions = [predict_sentiment(text) for text in sample_texts]

# Output results
print(f"Model Accuracy: {accuracy:.2f}")
print("\nSample Predictions:")
for text, pred in zip(sample_texts, predictions):
    print(f"Text: {text} => Sentiment: {pred}")
