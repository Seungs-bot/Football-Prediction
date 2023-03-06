import time
import pandas as pd
from functions import webscrape

def webscraping(setting = 'all'):
    data_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/data/"
    start_time = time.time()
    
    if setting == 'all':
        match_df = webscrape(year_start = 2016, year_end = 2022)
    
    #To update, read in the old csv and just update the most recent season data
    if setting == 'update':
        match_df_update = webscrape(year_start = 2021, year_end = 2022)
        match_df_update.to_csv(f'{data_filepath}matches_update.csv')
        match_df_update = pd.read_csv(f'{data_filepath}matches_update.csv', index_col = [0])
        match_df_to_update = pd.read_csv(f'{data_filepath}matches.csv', index_col = [0])
        match_df = match_df_to_update[match_df_to_update['season'] != 2022]
        match_df = pd.concat([match_df_update,match_df])

    match_df.to_csv(f"{data_filepath}matches.csv")
    end_time = time.time()
    print(f"FINISHED WEBSCRAPING: {round((end_time - start_time),2)} SECONDS")
    return

webscraping(setting = 'update')