from webscraping import webscraping
from feature_engineering import feature_engineering
from model_training import model_training
from hyperparameter_tuning import hyperparameter_tuning
from evaluate_model import evaluate_model
from make_predictions import make_predictions

def main(setting, season, matchweek):
    
    if setting == 'predict':
        feature_engineering(setting = "predict")
        make_predictions(season = season, matchweek = matchweek)
    
    else:

        if setting == 'cold start':
            webscraping(setting = "all")
        
        if setting == 'update':
            webscraping(setting = "update")

        feature_engineering(setting = "train")
        model_training(to_predict = "target")
        model_training(to_predict = "btts")
        model_training(to_predict = "above_2.5")
        hyperparameter_tuning(to_predict = "target")
        hyperparameter_tuning(to_predict = "btts")
        hyperparameter_tuning(to_predict = "above_2.5")
        evaluate_model(to_predict = 'target')
        evaluate_model(to_predict = 'btts')
        evaluate_model(to_predict = 'above_2.5')
        feature_engineering(setting = "predict")
        make_predictions(season = season, matchweek = matchweek)
        return
        
main(setting = 'update', season = 2022, matchweek = "Matchweek 24")