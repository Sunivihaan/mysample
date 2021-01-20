import pickle
import pandas as pd
from utils import *
from process import *
from collections import Counter
import argparse
import sys


def process_text(content, tfidf_vect):
	per_punct = count_punct(content) ** (1/4)
	per_cap = count_caps(content) ** (1/6)
	msg_length = len(content) ** (1/7)
	clean_content = clean_text(content)
	row = [msg_length, per_punct, per_cap]
	X_count = tfidf_vect.transform([clean_content])
	X_count = X_count.toarray()
	X_count = X_count[0].tolist()
	row.extend(X_count)
	return row


def predict_on_msg(args):

	with open(args.model_dir+"/"+args.tfidf_file,"rb") as fp:
		tfidf_vect = pickle.load(fp)

	with open(args.model_dir+"/"+args.features_file,"r") as fp:
		imp_features = json.load(fp)

	with open(args.model_dir+"/"+args.model_file,"rb") as fp:
		model = pickle.load(fp)	
		# model = fp.read()

	imp_features = imp_features.replace("[","").replace("]","").replace("'","").split(", ")
	features_df = process_text(args.msg, tfidf_vect)
	fdict = pd.DataFrame(np.array(features_df).reshape(1,-1), columns = imp_features)
	flag = model.predict(fdict)
	return flag[0]


def predict_from_api(msg):

	model_dir = "./models"
	tfidf_file = "tfidf_vect.pkl"
	features_file = "features_columns.json"
	model_file = "RandomForest_model.pkl"

	with open(model_dir+"/"+tfidf_file,"rb") as fp:
		tfidf_vect = pickle.load(fp)

	with open(model_dir+"/"+features_file,"r") as fp:
		imp_features = json.load(fp)

	with open(model_dir+"/"+model_file,"rb") as fp:
		model = pickle.load(fp)	

	imp_features = imp_features.replace("[","").replace("]","").replace("'","").split(", ")
	features_df = process_text(msg, tfidf_vect)
	fdict = pd.DataFrame(np.array(features_df).reshape(1,-1), columns = imp_features)
	flag = model.predict(fdict)
	return flag[0]

	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir",help="Model directory path")
	parser.add_argument("--msg",help="Input message")
	parser.add_argument("--tfidf_file",help="Tfidf filename",default="tfidf_vect.pkl")
	parser.add_argument("--features_file",help="Features filename",default="feature_columns.json")
	parser.add_argument("--model_file",help="Model filename",default=None)
	args = parser.parse_args()
	flag = predict_on_msg(args)
	# 0 is non-toxic and 1 is toxic
	if flag == 1:
		print("Message '{}' is toxic".format(args.msg))
	else:
		print("Message '{}' is non-toxic".format(args.msg))


if __name__ == "__main__":
	main()
