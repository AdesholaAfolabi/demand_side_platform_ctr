from __future__ import print_function
import post_process
import os
import json
import pickle
import io
import sys
import signal
import traceback
import yaml
import preprocess_data
import numpy as np
import json
import request
import flask
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import pickle
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


import tensorflow.keras.backend as K
#K.clear_session()

graph = tf.get_default_graph()


model_path = "/home/ec2-user/new_dsp_model/model/model.h5"
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    tf.keras.backend.set_session(session)
    wide_model = tf.keras.models.load_model(model_path)

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model(model_path) is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    global sess
    global graph
    import yaml
    config = yaml.safe_load(open("/home/ec2-user/new_dsp_model/features.yml"))
    FEATURE = config['features']
    data = None
    file =  flask.request.files["data"]
    if file.content_type == 'text/csv':
        data = flask.request.files["data"]
        data = pd.read_csv(data,usecols=FEATURE)
    elif file.content_type == 'application/json':
        data = flask.request.files["data"]
        data = pd.read_json(data)
        data=data[FEATURE]
        data['captured_time'] = data['captured_time'].astype(str)
    else:
        return flask.Response(response='This predictor only supports CSV/JSON data', status=414, mimetype='text/plain')

    dataframe = post_process.pipeline_object(data)
    with session.graph.as_default():
        tf.keras.backend.set_session(session)
        inf = wide_model.predict(dataframe)
    out = io.StringIO()
    pd.DataFrame({'results':inf.flatten()}).to_csv(out, index=False,sep=',',header=['score'])
    result = out.getvalue()
    return result

@app.route('/raw_requests', methods=['POST'])
def raw_request():
    global sess
    global graph
    import yaml
    config = yaml.safe_load(open("/home/ec2-user/new_dsp_model/features.yml"))
    FEATURE = config['features']
    df = None
    df = flask.request.get_json(force=True)
    data = pd.io.json.json_normalize(df)
    #data = pd.read_json(json.dumps(jsonStr))
    data=data[FEATURE]
    data['captured_time'] = data['captured_time'].astype(str)
    
    dataframe = post_process.pipeline_object(data)
    with session.graph.as_default():
        tf.keras.backend.set_session(session)
        inf = wide_model.predict(dataframe)
    out = io.StringIO()
    pd.DataFrame({'results':inf.flatten()}).to_csv(out, index=False,sep=',',header=['score'])
    result = out.getvalue()
    return result   
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
