# Machine-learning project
# Demand_Side_Platform_ctr

## Table of contents
* [General info](#general-info)
* [Expected Output]
* [Setup](#setup)
* [Pre and Post-Processing](#features)


## General info
This document provides instructions on how to build an api that can integrate a Click Prediction Service
into various solutions by making a request to the service's HTTP application endpoint.
The documentation is intended for developers/users who want to write applications that can
interact with the Click Prediction Service API. The service is REST adherent and can
be used with any programming language.

## Expected Output

The expected output of this application are probability scores of users which indicate their individual propensities to click on a digital campain Advertisement. This users are ranked on a scale of 0 - 1. Users with high probabilities have high propensity to click on the campaign adverts while users with low probabilities have low propensity. 


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

