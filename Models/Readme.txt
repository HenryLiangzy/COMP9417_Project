Files in folder:
=========================
best_<model_name>.joblib
    Learning model, model_name indicates the model used. Can be opened with python `joblib` library:
    with open (filepath, 'rb') as file:
        clr = file
        
    "clf" then can be used as trained model. Read "clf.named_steps" for details
=========================
model_collection.csv
    A csv file keeps record of optimal models
    
=========================
Readme.txt
    This file