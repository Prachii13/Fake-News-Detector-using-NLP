import sys
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

text = sys.argv[1]
vec = tfidf.transform([text])
pred = model.predict(vec)
print("✅ Prediction:", pred[0])
