# BTR Spark

Best Trip Recommender (BTR) Spark is a comprises the Spark job implementations of the steps performed by the BTR application, as explained below.

## Origin-Destination (OD) Matrix Estimation

This step consists of estimating the Origin-Destination (OD) Matrix for a Public Transportation System given Automatic Vehicle Location (AVL) and Automatic Fare Collection (AFC) data.

```
spark-submit <repo-folder-path>/od-builder.py <input-folder-path> <output-folder-path> <initial_date> <final_date>
```

## Training Data Preprocessing

This step consists of reorganizing the training data so that each record represents the trip from one stop to the next stop, thus increasing training data records and granularity to make models more robust. 

```
spark-submit <repo-folder-path>/data-preprocessing.py <btr-input-path> <btr-pre-processing-output-folder>
```

## Model Training

This step consists of training both duration and crowdedness prediction models using preprocessed training data. 

```
spark-submit <repo-folder-path>/model-training.py <training-data-path> <output-folder-path>
```

## Model Tunning

This step consists of tunning both duration and crowdedness prediction models using preprocessed training data to obtain the best model hyperparameters. 

```
spark-submit <repo-folder-path>/model-tunning.py <training-data-path> <btr2-tunning-output-path> <train-start-date(YYYY-MM-DD)> <train-end-date(YYYY-MM-DD)> <test-end-date(YYYY-MM-DD)>
```
