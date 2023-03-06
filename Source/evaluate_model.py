import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from functions import save_artefact, load_artefact
from classes import Football_Model, MissingDict

def evaluate_model(to_predict):
    model_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/models/"
    data_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/data/"
    start_time = time.time()
    print(f"STARTING {to_predict.upper()} MODEL EVALUATION...")

    #Load the football model class and all the necessary values
    fm = load_artefact(f"{model_filepath}{to_predict}_tuned_model.pickle")

    #Evaluate all the models on the combined objective score metric
    fm.evalaute_tuned_model_combined_objective()

    #Get predictions by the three models
    preds_rf, preds_logreg, preds_xgb = fm.get_predictions(data = fm.test)

    #Save the predictions into csv file
    preds_rf.to_csv(f"{data_filepath}{to_predict}_preds_rf.csv")
    preds_logreg.to_csv(f"{data_filepath}{to_predict}_preds_logreg.csv")
    preds_xgb.to_csv(f"{data_filepath}{to_predict}_preds_xgb.csv")

#evaluate_model('target')
#evaluate_model('btts')
#evaluate_model('above_2.5')