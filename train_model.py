import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os


os.makedirs('model', exist_ok=True)


data = {
    'resume_text': [
        "Experienced Python developer with Flask knowledge",
        "Skilled data analyst with Excel and Tableau experience",
        "Frontend developer with React and JavaScript",
        "Digital marketer with SEO and AdWords expertise"
    ],
    'label': ['Software Engineer', 'Data Analyst', 'Frontend Developer', 'Digital Marketer']
}

df = pd.DataFrame(data)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['resume_text'])


model = LogisticRegression()
model.fit(X, df['label'])


with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
