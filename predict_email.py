import joblib

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
feature_extraction = joblib.load('tfidf_vectorizer.pkl')


# Loop to classify multiple emails
while True:
    email_text = input("Enter email text (or type 'exit' to quit): ")
    if email_text.strip().lower() == 'exit':
        break
    if not email_text.strip():
        print("Please enter a non-empty email text.")
        continue
    features = feature_extraction.transform([email_text])
    prediction = model.predict(features)
    if prediction[0] == 1:
        print("It is a Spam mail")
    elif prediction[0] == 0:
        print("It is a Ham mail")
    else:
        print(f"Unknown prediction: {prediction[0]}")
