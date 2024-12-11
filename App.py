import pickle
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load the model
with open('D:\\Restaurant-Review-Using-NLP-main\\review_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('D:\\Restaurant-Review-Using-NLP-main\\Count_Vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        review = request.form['review']
        if review:
            review = re.sub('[^a-zA-Z]', ' ', review)
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            review = cv.transform([review])
            # Call your NLP model to get the sentiment prediction
            sentiment = model.predict(review)[0]
            result = "Good" if sentiment == 1 else "Bad"

    return render_template('index.html', result=result)
if __name__ == "__main__":
    app.run(debug = True, port=8000)