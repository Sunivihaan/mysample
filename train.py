import pickle
import glob
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import warnings
from sklearn.utils import shuffle
import string
import re
import argparse
import logging

import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from utils import *	
from process import *
warnings.filterwarnings("ignore")


def get_classifier(name):
	if name == "RandomForest":
		model = RandomForestClassifier()
	elif name == "SVM":
		model = SVC()
	elif name =="DecisionTree":
		model = DecisionTreeClassifier()
	elif name == "MultinomialNB":
		model = MultinomialNB()
	elif name == "LogisticRegression":
		model = LogisticRegression()
	return model

def test_classifiers(X_train_new, y_train_new, X_test, y_test, model_name):
	t = time.time()
	logging.info("Testing baseline classifier {}...".format(model_name))
	model = get_classifier(model_name)
	model.fit(X_train_new, y_train_new)
	y_train_pred = model.predict(X_train_new)
	get_model_statistics(y_train_new, y_train_pred, model_name, "train")
	y_pred = model.predict(X_test)
	get_model_statistics(y_test, y_pred, model_name, "test")
	logging.info("Time taken for {0} :- {1} ".format(model_name, (time.time()-t)/60))
	return

def select_best_model(X_train_new, y_train_new, X_test, y_test, model_name):
	model_params = {"RandomForest":{"n_estimators":[50,100,150, 200,300],"max_depth":[5, 8, 15, 20, 30, 50],"min_samples_split":[2,4,6, 8, 10]},
					"SVM":{},
					"DecisionTree":{},
					"LogisticRegression":{}}
	model = get_classifier(model_name)
	clf = GridSearchCV(model, model_params[model_name], cv = 10)
	grid = clf.fit(X_train_new, y_train_new)
	logging.info(grid.best_estimator_.get_params())
	return grid.best_estimator_
	
	


def build_model(args):

	t = time.time()
	data = get_processed_dataset(args, 6000, 5000, args.retrieve)
	data = data.dropna(axis = 0, how ='any') 
	print("Time taken :- ", time.time()-t/60, data.shape)
	logging.info("Time taken for obtaining processed dataset :- {}".format((time.time()-t)/60))
	logging.info("Shape of processed dataset :- {}".format(data.shape))
	
	target = data["msg_label"]
	features = data.drop("msg_label",axis=1)

	del data
	logging.info('{0}, {1}'.format(features.shape, target.shape))
	
	logging.info("Feature extraction ..........")
	X_train_new, y_train_new, X_test, y_test = feature_processing(features, target, args)

	logging.info("Building model ..........")

	if args.test_classifier:
		test_classifiers(X_train_new, y_train_new, X_test, y_test, args.model_name)

	elif args.perform_gridsearch:
		best_model = select_best_model(X_train_new, y_train_new, X_test, y_test, args.model_name)
		best_model_fit = best_model.fit(X_train_new, y_train_new)
		y_train_pred = best_model_fit.predict(X_train_new)
		get_model_statistics(y_train_new, y_train_pred, args.model_name, "train")
		y_pred = best_model_fit.predict(X_test)
		get_model_statistics(y_test, y_pred, args.model_name, "test")
	
	else:
		final_model = RandomForestClassifier(max_depth=50, n_estimators=100, min_samples_split=10)
		final_model.fit(X_train_new, y_train_new)
		y_train_pred = final_model.predict(X_train_new)
		get_model_statistics(y_train_new, y_train_pred, args.model_name, "train")
		y_pred = final_model.predict(X_test)
		get_model_statistics(y_test, y_pred, args.model_name, "test")

		with open(args.model_dir+"/"+args.model_file,"wb") as fp:
			pickle.dump(final_model,fp)


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path",help="Dataset file path", default="./data")
	parser.add_argument("--model_dir",help="Model directory path", default="./models")
	parser.add_argument("--model_name",help="Model name",default="SVM")
	parser.add_argument("--tfidf_file",help="Tfidf filename",default="tfidf_vect.pkl")
	parser.add_argument("--features_file",help="Features filename",default="feature_columns.json")
	parser.add_argument("--model_file",help="Model filename",default=None)
	parser.add_argument("--retrieve",help="Retrieve saved data or make new data",default=True, action='store_true')
	parser.add_argument("--logfile_name",help="Name of the log file",default="train.log")
	parser.add_argument("--test_classifier", help="Whether to test baseline models", default=False,action='store_true')
	parser.add_argument("--perform_gridsearch", help="Whether to do gridsearch", default=False,action='store_true')
	args = parser.parse_args()


	logging.basicConfig(filename='../Toxic_Analyzer_2/logs/'+args.logfile_name,
						level=logging.INFO,
						format='%(levelname)s: %(asctime)s %(message)s',
						datefmt='%m/%d/%Y %I:%M:%S')


	if args.model_file == None:
		args.model_file = '{}_model.pkl'.format(args.model_name)

	logging.info(args)
	if not os.path.exists(args.model_dir):
		os.mkdir(args.model_dir)
	
	build_model(args)

if __name__== "__main__":
	main()