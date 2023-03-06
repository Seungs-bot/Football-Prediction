import time
import pickle
import pandas as pd
from functions import get_predictors, save_artefact
from classes import Football_Model, MissingDict

def model_training(to_predict):
    model_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/models/"
    data_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/data/"
    start_time = time.time()
    print(f"STARTING INITIAL {to_predict.upper()} MODEL TRAINING...")
    matches_engineered = pd.read_csv(f'{data_filepath}matches_engineered.csv')
    match_result_predictors, above_goal_predictors, btts_predictors = get_predictors()
    
    #Logic to determine which variable to predict
    if to_predict == "target":
        predictors = match_result_predictors
    if to_predict == "btts":
        predictors = btts_predictors
    if to_predict == "above_2.5":
        predictors = above_goal_predictors

    #Mapping dictionary to standardize the team names
    map_values = {
        "Brighton and Hove Albion": "Brighton",
        "Leeds United": "Leeds",
        "Leiceister City": "Leiceister",
        "Manchester United": "Manchester Utd",
        "Newcastle United" : "Newcastle Utd",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
        "West Bromwich Albion" : "West Brom"
    }
    mapping = MissingDict(**map_values)
    
    #Initialize the Football Model class for model training and evaluation
    fm = Football_Model(data = matches_engineered, predictors = predictors, to_predict = to_predict, mapping = mapping)
    
    #Train and evaluate the initial model trained
    fm.train_initial_model()
    fm.evaluate_initial_model()
    
    #Evaluate the initial model trained on the combined objective
    fm.evaluate_initial_model_combined_objective()

    #Save the initial trained model
    output_model = f"{model_filepath}{to_predict}_initial_model.pickle"
    save_artefact(output_model, fm)

    end_time = time.time()
    print(f"FINISHED INITIAL {to_predict.upper()} MODEL TRAINING: {round((end_time - start_time),2)} SECONDS")
    return

#model_training('target')