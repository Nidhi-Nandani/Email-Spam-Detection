# Email Spam Detection

A machine learning project to classify emails as spam or not spam using Python. This project uses a dataset of emails, TF-IDF vectorization, and a trained classifier model to predict whether a given email is spam.

## Features
- Pre-trained spam classifier model
- TF-IDF vectorizer for email text
- Command-line prediction script
- Ready for integration with other applications

## Project Structure
```
email_classifier.py         # Model training and evaluation script
predict_email.py           # Script to predict if an email is spam
mail_dataset.csv           # Dataset of emails (not included in repo due to size)
spam_classifier_model.pkl  # Trained model (not included in repo due to size)
tfidf_vectorizer.pkl       # TF-IDF vectorizer (not included in repo due to size)
README.md                  # Project documentation
```

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/Nidhi-Nandani/Email-Spam-Detection.git
   ```
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the prediction script:
   ```
   python predict_email.py
   ```

## Notes
- Large files (dataset, model, vectorizer) are not included in the repository. Please contact the author if you need access.
- For best results, use Python 3.7 or higher.

## Author
- Nidhi Nandani
- Email: nidhinandani001@gmail.com

## License
This project is licensed under the MIT License.
