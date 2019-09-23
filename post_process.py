import numpy as np
from sklearn.pipeline import Pipeline
import preprocess_data
import tensorflow as tf

import pickle

pipeline_path = "/home/ec2-user/new_dsp_model/model/pipeline.pkl"

def pipeline_object(data):
    pickle_in = open(pipeline_path, "rb")
    full_pipeline = pickle.load(pickle_in)
    predict_data = preprocess_data.preprocessing_file(data)
    clean_data = full_pipeline.transform(predict_data)
    print (clean_data.shape)
    return clean_data
