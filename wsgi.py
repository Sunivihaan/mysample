import flask
from flask import Flask ,request, jsonify
from predict import *

application = Flask(__name__)

@application.route("/")
def hello():
	#print(os.environ)
	return "Please Type a Message!"

@application.route('/predict_msg', methods=['POST'])
def predict_msg():
    param = request.get_json(force=True)
    msg = str(param['query'])
    flag = str(predict_from_api(msg))
    result = {"message":msg, "output":flag}
    return jsonify(result)

#checking in local environment
if __name__ == "__main__":
	application.run()

