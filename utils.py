import re
import string
import pickle

import nltk
import logging

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, precision_score, recall_score, f1_score

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

# 1. Lowercase text
# 2. Remove whitespace
# 3. Remove numbers
# 4. Remove special characters
# 5. Remove emails
# 6. Remove stop words
# 7. Remove NAN
# 8. Remove weblinks
# 9. Expand contractions (if possible not necessary)
# 10. Tokenize

def clean_text(sentence):
	sentence=str(sentence)
	sentence = sentence.lower()
	sentence=sentence.replace('{html}',"")
	sentence = sentence.replace("_","") 
	cleanr = re.compile('<.*?>')
	cleantext = re.sub(cleanr, '', sentence)
	rem_url=re.sub(r'http\S+', '',cleantext)
	rem_num = re.sub('[0-9]+', '', rem_url)
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(rem_num)  
	filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
	stem_words = [stemmer.stem(w) for w in filtered_words]
	# lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
	return " ".join(stem_words)


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

def lemmatize(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

def count_caps(text):
    count = sum([1 for char in text if char in string.printable[36:62]])
    return round(count/(len(text) - text.count(" ")), 3)*100


def get_model_statistics(y_test, y_pred, model_name, type='train'):
	logging.info('Stats for model {}'.format(model_name))
	logging.info('{0} Confusion matrix {1} '.format(type, confusion_matrix(y_test,y_pred)))
	logging.info('{0} Classification report {1} '.format(type, classification_report(y_test,y_pred)))
	logging.info('{0} Accuracy score {1} '.format(type, accuracy_score(y_test,y_pred)))
	logging.info('{0} ROC AUC score {1} '.format(type, roc_auc_score(y_test,y_pred)))
	return 