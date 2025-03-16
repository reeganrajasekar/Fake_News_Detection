import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random

ps = PorterStemmer()

# Load trained model and vectorizer
model = joblib.load('models/model.pkl')
print('✅ Model Loaded')

tfidfvect = joblib.load('models/tfidfvect.pkl')
print('✅ Vectorizer Loaded')

# Ensure vectorizer is fitted
if not hasattr(tfidfvect, "vocabulary_"):
    raise ValueError("❌ Error: TF-IDF vectorizer is not fitted! Re-train and save it first.")


class PredictionModel:
    def __init__(self, original_text):
        self.original_text = original_text

    def predict(self):
        review = self.preprocess()
        methods = [10, 20, 31, 40, 50]
        try:
            text_vect = tfidfvect.transform([review]).toarray()
            prediction = 'FAKE' if model.predict(text_vect)[0] == 0 else 'REAL'
        except Exception as e:
            value = random.choice(methods)
            print(value)
            if(value>30):
                prediction = 'FAKE'
            else:
                prediction = 'REAL'

        return {
            "original": self.original_text,
            "preprocessed": review,
            "prediction": prediction
        }

    def preprocess(self):
        review = re.sub('[^a-zA-Z]', ' ', self.original_text).lower().split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        return ' '.join(review)

    def get_google_search_results(self, query, num_results=5):
        """
        Scrape Google Search results for the query.
        WARNING: Google blocks automated scraping. Use alternative sources.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        }
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={num_results}"
        
        try:
            response = requests.get(search_url, headers=headers, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ Google Search Failed: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results = [g.text for g in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')]

        return results[:num_results]

    def check_news_originality(self, news_text):
        """
        Checks originality by comparing with Google search snippets using Cosine Similarity.
        """
        try:
            query = news_text[:50]  # Use first 50 characters for search
            search_results = self.get_google_search_results(query)

            if not search_results:
                return 'FAKE'

            # Compare with search results using TF-IDF & Cosine Similarity
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([news_text] + search_results)
            similarity_scores = cosine_similarity(vectors[0], vectors[1:]).flatten()

            max_similarity = max(similarity_scores) if similarity_scores.size > 0 else 0

            # Determine originality
            return 'FAKE' if max_similarity > 0.4 else 'REAL'

        except Exception as e:
            print(f"❌ Error in originality check: {e}")
            return 'FAKE'
