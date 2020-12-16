import joblib
import pandas as pd

class Classifier(object):
    def __init__(self):
        self.vectorizer = joblib.load("movie_vectorizer_dump.pkl")
        self.model = joblib.load("movie_model_dump.pkl")
        self.mlb = joblib.load("movie_mlb_dump.pkl")
        self.treshold_df = pd.read_csv('treshold_df.csv',index_col='index')

    def get_name_by_label(self, label):
        try:
            return  label
        except:
            return "label error"

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            preds = self.model.predict_proba(vectorized)
            for n, yi in enumerate(['action', 'adventure', 'animation', 'biography', 'comedy', 'crime','drama', 'family', 'fantasy', 'history', 'horror', 'music', 'musical','mystery', 'romance', 'sci-fi', 'sport', 'thriller', 'war', 'western']):
                preds[:, n] = preds[:, n] > self.treshold_df.loc[yi, 'best_treshold']
            if preds.sum() == 0:
                preds[6] = 1
            return ' '.join(self.mlb.inverse_transform(preds)[0])
        except:
            print("prediction error")
            return None 

    def get_result_message(self, text):
        prediction = self.predict_text(text)
        print(prediction)
        return self.get_name_by_label(prediction)
