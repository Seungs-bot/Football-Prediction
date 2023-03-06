import time
import pandas as pd
import numpy as np
from functions import rolling_averages

def feature_engineering(setting = 'train'):
    data_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/data/"

    start_time = time.time()
    print("STARTING FEATURE ENGINEERING...")

    if setting == 'train':
        matches = pd.read_csv(f'{data_filepath}matches.csv', index_col=0)

    if setting == 'predict':
        matches = pd.read_csv(f'{data_filepath}matches.csv', index_col=0)
        matches_to_predict = pd.read_csv(f'{data_filepath}matches_to_predict.csv')
        matches = pd.concat([matches, matches_to_predict])
        matches = matches[matches['comp'] == 'Premier League']

    #Creating date as a datetime variable
    matches["date"] = pd.to_datetime(matches["date"])

    #Creating venue codes as a categorical variable
    matches["venue_code"] = matches["venue"].astype("category").cat.codes

    #Creating opponent codes as a categorical variable
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes

    #Creating team codes as a categorical variable
    matches["team_code"] = matches["team"].astype("category").cat.codes

    #Keeping the hour as integer value
    matches["hour"] = matches["time"].str.replace(":.+","", regex = True).astype("int")

    #Creating day codes as a categorical variable
    matches["day_code"] = matches["date"].dt.dayofweek

    #Creating numerical variable for team's form
    form_dict = {'W':1, 'D':0, 'L':-1}
    matches["form"] = matches["result"].replace(form_dict)

    #Creating a target variable
    matches["target"] = (matches["result"] == "W").astype("int")
    #matches["target"] = matches["result"].astype("category").cat.codes

    #Creating a team to score variable
    matches["tts"] = np.where(matches["gf"] > 0, 1, 0)
    matches["tts"] = matches["tts"].astype("category")

    #Creating a team to concede variable
    matches["ttc"] = np.where(matches["ga"] > 0, 1, 0)
    matches["ttc"] = matches["ttc"].astype("category")

    #Creating a both team to score variable
    matches["btts"] = np.where(((matches["ga"] > 0) & (matches["gf"] > 0)), 1, 0)
    matches["btts"] = matches["btts"].astype("category")

    #Creating a goal above 2.5 variable
    matches["above_2.5"] = np.where((matches["ga"] + matches["gf"]) > 2, 1, 0)
    matches["above_2.5"] = matches["above_2.5"].astype("category")

    #Creating a goal above 3.5 variable
    matches["above_3.5"] = np.where((matches["ga"] + matches["gf"]) > 3, 1, 0)
    matches["above_3.5"] = matches["above_3.5"].astype("category")

    #Predictors for match results
    predictors = ["venue_code", "opp_code", "team_code"]
    cols_form = ["gf", "ga", "sh", "sot", "dist", "form", "xg", "poss", "sota", "save%", "cs", "psxg", "cmp", "cmp%", "prgdist",
                "ast", "ppa", "prgp", "sca", "gca", "tklw", "int", "tkl+int", "err", "succ", "succ%", "crdy", "fls", "won%"]
    cols_avg = ["gf", "ga", "form", "xg", "xga", "poss", "cs"]
    new_cols_form = [f"{c}_rolling_3" for c in cols_form]
    new_cols_avg = [f"{c}_rolling_365" for c in cols_avg]

    #Finding rolling average for the team for every 40 matches
    #This will be able to show how the team is expected to perform over a long period of time
    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols_avg, new_cols_avg, 40))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])

    #Find rolling averages for the team for every 3 matches
    #This will be able to show the recent form of the team in the short term
    matches_rolling = matches_rolling.groupby("team").apply(lambda x: rolling_averages(x, cols_form, new_cols_form, 3))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    
    if setting == 'train':
        matches_rolling.to_csv(f"{data_filepath}matches_engineered.csv")

    if setting == 'predict':
        matches_rolling.to_csv(f"{data_filepath}matches_to_predict_engineered.csv")
    
    end_time = time.time()
    print(f"FINISHED FEATURE ENGINEERING: {round((end_time - start_time),2)} SECONDS")
    return

#feature_engineering(setting = 'predict')