# demand_side_platform_ctr
# Machine-learning project
# DSP CTR MODEL API

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Screenshots](#screenshots)
* [Pre and Post-Processing](#features)


## General info
This document provides instructions on how to build an api that can integrate a Click Prediction Service
into various solutions by making a request to the service's HTTP application endpoint.
The documentation is intended for developers/users who want to write applications that can
interact with the Click Prediction Service API. The service is REST adherent and can
be used with any programming language.

## Screenshots
![Example screenshot](./img/screenshot.png)

## Setup
Set up the flask application environment. In the same environment, there is a `Preprocessing.py` and `Postprocess.py` scripts. The `Preprocessing.py` does the data cleaning and preprocessing. The `Postprocess.py` loads the preprocessor pikle object which transforms the data. The model folder contains a `model.h5` and the `preprocessor.pkl` object. The `Features.yml` contains features used for building the model. 
 
 
    Flask
            |__ App.py
            |__ Features.yml
            |__ Preprocessing.py
            |__ Postprocess.py
            |__ Model
              |__model.h5
              |__ preprocessor.pkl
   
        
## Pre/Post-Processing
Flask TensorFlow Serving supports the following Content-Types for requests:
*application/json (default)
*text/csv
 
The flask app will convert data in these formats to TensorFlow Serving REST API requests, and will send these requests to the default serving signature of our Saved Model. Read in the yaml file but you need to specify the path to your yaml file.
```ruby
import yaml
    config = yaml.safe_load(open("add_your_path/features.yml"))
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
After reading in the data, the pipeline_object function is called on the data. Which preprocesses the data and fits the data in to a pipeline object before making predictions.
ruby
dataframe = post_process.pipeline_object(data)
    with session.graph.as_default(): 
        tf.keras.backend.set_session(session)
        inf = wide_model.predict(dataframe)
    out = io.StringIO()
    pd.DataFrame({'results':inf.flatten()}).to_csv(out, index=False,sep=',',header=['score'])
    result = out.getvalue()
    return result
You can set host and port of the app
ruby
if __name__ == "__main__":
    app.run(host="0.0.0.0", port= ----, debug=True)
