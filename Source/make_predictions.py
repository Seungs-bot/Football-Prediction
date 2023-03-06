import time
import pickle
import pandas as pd
import numpy as np
from functions import save_artefact, load_artefact, set_result_predictions, set_btts_predictions, set_above_predictions, merge_all
from classes import Football_Model, MissingDict

def make_predictions(season, matchweek):
    data_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/data/"
    model_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/models/"
    prediction_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/predictions/"
    matches_to_predict_engineered = pd.read_csv(f"{data_filepath}matches_to_predict_engineered.csv")
    matches_to_predict = matches_to_predict_engineered[matches_to_predict_engineered["round"] == matchweek]
    matches_to_predict = matches_to_predict[matches_to_predict["season"] == season]

    #Get all the tuned models and update the data for the different betting types
    fm_target = load_artefact(f"{model_filepath}target_tuned_model.pickle")
    fm_target.update_data(matches_to_predict_engineered)
    
    fm_btts = load_artefact(f"{model_filepath}btts_tuned_model.pickle")
    fm_btts.update_data(matches_to_predict_engineered)
    
    fm_above = load_artefact(f"{model_filepath}above_2.5_tuned_model.pickle")
    fm_above.update_data(matches_to_predict_engineered)

    #Get all the predictions by all models for the different betting types
    target_preds_rf, target_preds_logreg, target_preds_xgb = fm_target.get_predictions(data=matches_to_predict)
    btts_preds_rf, btts_preds_logreg, btts_preds_xgb = fm_btts.get_predictions(data=matches_to_predict)
    above_preds_rf, above_preds_logreg, above_preds_xgb = fm_above.get_predictions(data=matches_to_predict)

    #Set the predictions and merge them all for all the three different model types
    
    #RANDOM FOREST CLASSIFIER
    target_preds_rf = set_result_predictions(target_preds_rf)
    btts_preds_rf = set_btts_predictions(btts_preds_rf)
    above_preds_rf = set_above_predictions(above_preds_rf)
    preds_rf = merge_all(target_preds_rf, btts_preds_rf, above_preds_rf)
    preds_rf = preds_rf[['date', 'round', 'x', 'y', 'result_preds', 'btts_preds', 'above_preds']]

    #LOGISTIC REGRESSION
    target_preds_logreg = set_result_predictions(target_preds_logreg)
    btts_preds_logreg = set_btts_predictions(btts_preds_logreg)
    above_preds_logreg = set_above_predictions(above_preds_logreg)
    preds_logreg = merge_all(target_preds_logreg, btts_preds_logreg, above_preds_logreg)
    preds_logreg = preds_logreg[['date', 'round', 'x', 'y', 'result_preds', 'btts_preds', 'above_preds']]

    #XGBOOST CLASSIFIER
    target_preds_xgb = set_result_predictions(target_preds_xgb)
    btts_preds_xgb = set_btts_predictions(btts_preds_xgb)
    above_preds_xgb = set_above_predictions(above_preds_xgb)
    preds_xgb = merge_all(target_preds_xgb, btts_preds_xgb, above_preds_xgb)
    preds_xgb = preds_xgb[['date', 'round', 'x', 'y', 'result_preds', 'btts_preds', 'above_preds']]

    #Save all the predictions to file
    preds_rf.to_csv(f'{prediction_filepath}RF PREDICTIONS.csv')
    preds_logreg.to_csv(f'{prediction_filepath}LOGREG PREDICTIONS.csv')
    preds_xgb.to_csv(f'{prediction_filepath}XGBOOST PREDICTIONS.csv')

#make_predictions(season = 2022, matchweek = 'Matchweek 20')