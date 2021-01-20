import glob
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
import string
import logging
from itertools import islice
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from utils import *

def get_dataset(filename):
	logging.info("Get dataset .................")
	data = pd.read_csv(filename)
	data = data.drop("Unnamed: 0",axis=1)
	data.rename(columns={'length': 'msg_length','content':'msg_content','label':'msg_label'}, inplace=True)
	data = shuffle(data).reset_index(drop=True)
	data = data[1000:]
	return data

def get_processed_dataset(args, non_toxic_count, toxic_count, retrieve=False):

	save_data_path = args.data_path.split("/")[1]+"/"+"processed_data.csv"
	if retrieve == False:
		data = get_dataset(args.data_path)
		logging.info("Retrieved raw dataset")
		# Pick comments with length less than the mean length
		data = data[data["msg_length"]<=data["msg_length"].mean()]

		# Get new features - percentage of capital letters and percentage of punctuation in a comment
		data["per_punct"] = data["msg_content"].apply(lambda x: count_punct(x))
		data["per_cap"] = data["msg_content"].apply(lambda x: count_caps(x))

		# Transform new features
		data["msg_length"] = data["msg_length"] ** (1/7)
		data["per_punct"] = data["per_punct"] ** (1/4)
		data["per_cap"] = data["per_cap"] ** (1/6)

		# Clean comments
		data["clean_content"] = data["msg_content"].apply(lambda s:clean_text(s))
		data = data.dropna(axis = 0, how ='any')  
		# Drop original comments
		data = data.drop("msg_content",axis=1)
		data["msg_label"] = data["msg_label"].astype(np.int64)

		# Pick non-toxic and toxic messages
		non_toxic = shuffle(data[data["msg_label"]==0]).reset_index(drop=True)[0:non_toxic_count]
		toxic = shuffle(data[data["msg_label"]==1]).reset_index(drop=True)[0:toxic_count]
		data = pd.concat([toxic, non_toxic])
		data.to_csv(save_data_path,index=False)	
		logging.info("Dataset saved")
	else:
		
		data = pd.read_csv(save_data_path)
		logging.info("Saved dataset retrieved")
	return data

def feature_processing(features, target, args):

	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
	logging.info('{0}, {1}, {2}, {3} '.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

	tfidf_vect = TfidfVectorizer(min_df=0.001)
	tf_fit = tfidf_vect.fit(X_train["clean_content"])

	with open(args.model_dir+"/"+args.tfidf_file,"wb") as fp:
		pickle.dump(tf_fit, fp)

	vocabulary = tfidf_vect.get_feature_names()

	X_train_features = tf_fit.transform(X_train["clean_content"])
	X_train_features = pd.DataFrame(X_train_features.toarray(),columns=vocabulary)
	X_train = pd.concat([X_train[["msg_length","per_punct","per_cap"]].reset_index(drop=True), X_train_features],axis=1)

	X_test_features = tf_fit.transform(X_test["clean_content"])
	X_test_features = pd.DataFrame(X_test_features.toarray(),columns=vocabulary)
	X_test = pd.concat([X_test[["msg_length","per_punct","per_cap"]].reset_index(drop=True), X_test_features],axis=1)

	logging.info('{0}, {1}, {2}, {3} '.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

	sm = SMOTE(random_state = 33)
	X_train_new, y_train_new = sm.fit_sample(X_train, y_train)
	logging.info('{0}, {1}, {2}, {3} '.format(X_train_new.shape, y_train_new.shape, X_test.shape, y_test.shape))

	with open(args.model_dir+"/"+args.features_file,"w") as fp:
		json.dump(str(list(X_train_new.columns)),fp)

	return X_train_new, y_train_new, X_test, y_test