import pickle


class ProductClassifier:

    def __init__(self):
        self.vectorizer = None
        self.clf = None
        self.pca = None

    @staticmethod
    def preprocess_text(text: str, stop_symbols=None):
        text = text.lower()
        if not stop_symbols:
            stop_symbols = {'\n', ',', '-'}
        return ''.join([i[0] for i in zip(text, text[1:] + '#') if i[0] != i[1] and i[0] not in stop_symbols])

    def predict(self, ocr_result):
        ocr_result = ''.join([self.preprocess_text(i[1]) for i in ocr_result])
        vec = self.vectorizer.transform([ocr_result])
        x_embedded = self.pca.transform(vec.toarray())
        probabilities = self.clf.predict_proba(x_embedded)
        return self.clf.classes_[probabilities.argmax()], probabilities

    def load(self, file_path):
        pc = pickle.load(open(file_path, 'rb'))
        self.vectorizer = pc.vectorizer
        self.clf = pc.clf
        self.pca = pc.pca

    def save(self, file_path):
        pickle.dump(self, open(file_path, 'wb'))
