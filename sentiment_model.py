import glove_embedding
from embedding_manager_cyd import EmbeddingManagerCyd
from embedding_manager_cyd import Embedding_Type
from sklearn.externals import joblib #jbolib模块
from nltk import pos_tag, word_tokenize
import utils
from utils import handle_text
import numpy as np

class SentimentModel(object):
    def __init__(self):
        # self.model = joblib.load('model/svm_clf_used.pkl')
        self.model = joblib.load('model/svm_clf.pkl')
        self.embeddingManagerCyd = EmbeddingManagerCyd()

    def predict_prob(self, review_segment):
        vectors = [self.embeddingManagerCyd.getEmbedding(review_segment, Embedding_Type.glove, True, False)]
        features = np.array(vectors, dtype=np.float16)
        proba_value = self.model.predict_proba(features)
        score = proba_value[:, 1]
        return np.float16(score[0])




