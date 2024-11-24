import gensim.downloader as api
import nltk

from preprocessings.basic_and_additionnal_preprocessing import basic_and_additionnal_preprocessing
from preprocessings.basic_preprocessing import basic_preprocessig

nltk.download('stopwords')
nltk.download('wordnet')
embeddings_model = api.load("glove-twitter-200")


basic_and_additionnal_preprocessing("train_tweets",embeddings_model)
basic_and_additionnal_preprocessing("eval_tweets",embeddings_model)


basic_preprocessig("train_tweets",embeddings_model)
basic_preprocessig("eval_tweets",embeddings_model)