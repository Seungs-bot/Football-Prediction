import time
import pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def rolling_averages(group, cols, new_cols, n):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(n, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def get_predictors():
    #Predictors for match results and above 2.5/3.5
    predictors = ["venue_code", "opp_code", "team_code"]
    cols_form = ["gf", "ga", "sh", "sot", "dist", "form", "xg", "poss", "sota", "save%", "cs", "psxg", "cmp", "cmp%", "prgdist",
                "ast", "ppa", "prog", "sca", "gca", "tklw", "int", "tkl+int", "err", "succ", "succ%", "crdy", "fls", "won%"]
    cols_avg = ["gf", "ga", "form", "xg", "xga", "poss", "cs"]
    new_cols_form = [f"{c}_rolling_3" for c in cols_form]
    new_cols_avg = [f"{c}_rolling_365" for c in cols_avg]
    predictors1 = predictors + new_cols_form + new_cols_avg

    #Predictors for both team to score
    predictors = ["venue_code", "opp_code", "team_code"]
    cols_form = ["gf", "sh", "sot", "dist", "form", "xg", "poss", "sota", "cmp", "cmp%", "prgdist", "ast", "ppa", 
                "prog", "sca", "gca", "succ", "succ%", "crdy", "fls", "won%"]
    cols_avg = ["gf", "ga", "form", "xg", "xga", "poss", "cs"]
    new_cols_form = [f"{c}_rolling_3" for c in cols_form]
    new_cols_avg = [f"{c}_rolling_365" for c in cols_avg]
    predictors2 = predictors + new_cols_form + new_cols_avg

    return predictors1, predictors1, predictors2

def save_artefact(filepath, data):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    pkl = open(filepath, "wb")
    pickle.dump(data, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    pkl.close()

def load_artefact(filepath):
    pkl = open(filepath, "rb")
    res = pickle.load(pkl)
    pkl.close()
    return res

def webscrape(year_start, year_end):
    years = list(range(year_end,year_start,-1))
    print(f"STARTING WEBSCRAPING FROM {years[0]} TO {years[-1]}...")
    standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    all_matches = []

    for year in years:
        print(f"Scraping match data from the {year} season...")
        data = requests.get(standings_url)
        soup = BeautifulSoup(data.text, features="lxml")
        standings_table = soup.select('table.stats_table')[0]

        links = [l.get("href") for l in standings_table.find_all('a')]
        links = [l for l in links if '/squads/' in l]
        team_urls = [f"https://fbref.com{l}" for l in links]
        
        previous_season = soup.select("a.prev")[0].get("href")
        standings_url = f"https://fbref.com{previous_season}"
        time.sleep(5)
        
        for team_url in team_urls:
            team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
            print(f"   Getting {team_name} data...", end="")
            data = requests.get(team_url)
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
            soup = BeautifulSoup(data.text, features="lxml")
            stat_links = [l.get("href") for l in soup.find_all('a')]
            
            #Scrape the team shooting data
            links = [l for l in stat_links if l and 'all_comps/shooting/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            df = pd.read_html(data.text, match="Shooting")[0]
            df.columns = df.columns.droplevel()
            
            try:
                team_data = matches.merge(df[["Date", "Gls", "Sh", "SoT", "Dist"]], on="Date")
            except ValueError:
                continue
            
            #print("shooting, ",end="")
            time.sleep(5)
            
            #Scrape the team keeping data
            links = [l for l in stat_links if l and 'all_comps/keeper/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            df = pd.read_html(data.text, match="Goalkeeping")[0]
            df.columns = df.columns.droplevel()
            
            try:
                team_data = team_data.merge(df[["Date", "SoTA", "Save%", "CS", "PSxG"]], on="Date")
            except ValueError:
                continue
            
            #print("goalkeeping, ",end="")
            time.sleep(5)
            
            #Scrape the team passing data
            links = [l for l in stat_links if l and 'all_comps/passing/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            df = pd.read_html(data.text, match="Passing")[0]
            df.columns = df.columns.droplevel()
            
            try:
                team_data = team_data.merge(df[["Date", "Cmp", "Cmp%", "PrgDist", "Ast", "PPA", "Prog"]], on="Date")
            except ValueError:
                continue
            
            #print("passing, ",end="")
            time.sleep(5)
            
            #Scrape the team goal and shot creation data
            links = [l for l in stat_links if l and 'all_comps/gca/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            df = pd.read_html(data.text, match="Goal and Shot Creation")[0]
            df.columns = df.columns.droplevel()
            
            try:
                team_data = team_data.merge(df[["Date", "SCA", "GCA"]], on="Date")
            except ValueError:
                continue
            
            #print("gca, ",end="")
            time.sleep(5)
            
            #Scrape the team defensive data
            links = [l for l in stat_links if l and 'all_comps/defense/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            df = pd.read_html(data.text, match="Defensive Actions")[0]
            df.columns = df.columns.droplevel()
            
            try:
                team_data = team_data.merge(df[["Date", "TklW", "Int", "Tkl+Int", "Err"]], on="Date")
            except ValueError:
                continue
            
            #print("defense, ",end="")
            time.sleep(5)
            
            #Scrape the team possession data
            links = [l for l in stat_links if l and 'all_comps/possession/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            df = pd.read_html(data.text, match="Possession")[0]
            df.columns = df.columns.droplevel()
            
            try:
                team_data = team_data.merge(df[["Date", "Succ", "Succ%"]], on="Date")
            except ValueError:
                continue
            
            #print("possession, ",end="")
            time.sleep(5)
            
            #Scrape the team miscellaneous data
            links = [l for l in stat_links if l and 'all_comps/misc/' in l]
            data = requests.get(f"https://fbref.com{links[0]}")
            df = pd.read_html(data.text, match="Miscellaneous Stats")[0]
            df.columns = df.columns.droplevel()
            
            try:
                team_data = team_data.merge(df[["Date", "CrdY", "Fls", "Won%"]], on="Date")
            except ValueError:
                continue
            
            #print("misc_stats, ",end="")
            time.sleep(5)
            
            team_data = team_data[team_data["Comp"] == "Premier League"]
            team_data["Season"] = year
            team_data["Team"] = team_name
            all_matches.append(team_data)
            print(" DONE!")

    match_df = pd.concat(all_matches)
    match_df.columns = [c.lower() for c in match_df.columns]
    return match_df


#==========================================================================================#
#                            FUNCTIONS USED FOR DATA ENGINEERING                           #
#==========================================================================================#

#Creating a result prediction and result actual column
def set_result_predictions(df):
    df['result_preds'] = df['actual_x']
    df['result_actual'] = df['actual_x']
    
    #Setting predictions
    df.loc[((df['predicted_x'] == 1) & (df['predicted_y'] == 0)), ['result_preds']] = 'X'
    df.loc[((df['predicted_x'] == 0) & (df['predicted_y'] == 1)), ['result_preds']] = 'Y'
    df.loc[((df['predicted_x'] == 0) & (df['predicted_y'] == 0)), ['result_preds']] = 'D'
    df.loc[((df['predicted_x'] == 1) & (df['predicted_y'] == 1)), ['result_preds']] = 'NA'
    
    #Setting results
    df.loc[(df['actual_x'] == 1) , ['result_actual']] = 'X'
    df.loc[(df['actual_y'] == 1) , ['result_actual']] = 'Y'
    df.loc[((df['actual_x'] == 0) & (df['actual_y'] == 0)), ['result_actual']] = 'D'
    
    return df

#Creating a both teams to score prediction and both teams to score actual column
def set_btts_predictions(df):
    df['btts_preds'] = df['actual_x']
    df['btts_actual'] = df['actual_x']
    
    #Setting predictions
    df.loc[((df['predicted_x'] == 1) & (df['predicted_y'] == 1)), ['btts_preds']] = 'Y'
    df.loc[((df['predicted_x'] == 0) & (df['predicted_y'] == 0)), ['btts_preds']] = 'N'
    df.loc[((df['predicted_x'] == 1) & (df['predicted_y'] == 0)), ['btts_preds']] = 'NA'
    df.loc[((df['predicted_x'] == 0) & (df['predicted_y'] == 1)), ['btts_preds']] = 'NA'
    
    #Setting results
    df.loc[(df['actual_x'] == 1) , ['btts_actual']] = 'Y'
    df.loc[(df['actual_x'] == 0) , ['btts_actual']] = 'N'
    
    return df

#Creating a above_2.5 prediction and below_2.5 actual column
def set_above_predictions(df):
    df['above_preds']= df['actual_x']
    df['above_actual'] = df['actual_x']
    
    #Setting predictions
    df.loc[((df['predicted_x'] == 1) & (df['predicted_y'] == 1)), ['above_preds']] = 'Y'
    df.loc[((df['predicted_x'] == 0) & (df['predicted_y'] == 0)), ['above_preds']] = 'N'
    df.loc[((df['predicted_x'] == 1) & (df['predicted_y'] == 0)), ['above_preds']] = 'NA'
    df.loc[((df['predicted_x'] == 0) & (df['predicted_y'] == 1)), ['above_preds']] = 'NA'
    
    #Setting results
    df.loc[(df['actual_x'] == 1) , ['above_actual']] = 'Y'
    df.loc[(df['actual_x'] == 0) , ['above_actual']] = 'N'
    
    return df

#Create a new dataframe integrating all three predictions and actual outcomes
def merge_all(target_df, btts_df, above_df):
    df = target_df.merge(btts_df[['btts_preds','btts_actual']], left_index = True, right_index = True)
    df = df.merge(above_df[['above_preds','above_actual']], left_index = True, right_index = True)
    df['x'] = df['team_x']
    df['y'] = df['team_y']
    df['round'] = df['round_x']
    df = df[["date","round","x","y","result_preds","result_actual","btts_preds","btts_actual","above_preds","above_actual"]]
    return df

#==========================================================================================#
#                 FUNCTIONS USED FOR CALCULATING ODDS AND EXPECTED RETURNDS                #
#==========================================================================================#

#Calculate the objective scores to find the odds
def calculate_betting_odds(df_target, df_btts, df_above):
    
    #Calculating betting odds for a team to win
    win_count = df_target[(df_target["predicted_x"] == 1) & (df_target["predicted_y"] == 0)]
    precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)
    win_odd = round(1/precision_win, 3)

    #Calculating betting odds for teams to draw
    draw_count = df_target[(df_target["predicted_x"] == 0) & (df_target["predicted_y"] == 0)]
    precision_draw = len(draw_count[draw_count['result_x'] == 'D']) / len(draw_count)
    draw_odd = round(1/precision_draw, 3)
    
    #Calculating betting odds for both teams to score
    btts_count = df_btts[(df_btts["predicted_x"] == 1) & (df_btts["predicted_y"] == 1)]
    precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)
    btts_odd = round(1/precision_btts, 3)
    
    #Calculating betting odds for not both teams to score
    nbtts_count = df_btts[(df_btts["predicted_x"] == 0) & (df_btts["predicted_y"] == 0)]
    precision_nbtts = len(nbtts_count[(nbtts_count['actual_x'] == 0) & (nbtts_count['actual_y'] == 0)]) / len(btts_count)
    nbtts_odd = round(1/precision_btts, 3)
    
    #Calculating betting odds for above 2.5 goals
    above_count = df_above[(df_above["predicted_x"] == 1) & (df_above["predicted_y"] == 1)]
    precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)
    above_odd = round(1/precision_above, 3)

    #Calculating betting odds for below 2.5 goals
    below_count = df_above[(df_above["predicted_x"] == 0) & (df_above["predicted_y"] == 0)]
    precision_below = len(below_count[below_count["actual_x"] == 0]) / len(below_count)
    below_odd = round(1/precision_below, 3)
    
    #Return all the betting odds
    return win_odd, draw_odd, btts_odd, nbtts_odd, above_odd, below_odd

def calculate_results(df, bet_amount, win_odd, draw_odd, btts_odd, nbtts_odd, above_odd, below_odd, mos=0):
    #Create new return column to contain all the expected returns and set column types
    total_expected_profits = []
    result_expected_profits = []
    btts_expected_profits = []
    above_expected_profits = []
    total_bet_num = 0
    result_bet_num = 0
    btts_bet_num = 0
    above_bet_num = 0
    
    #Calculate the new odds using the margin of safety
    win_odd += mos
    draw_odd += mos
    btts_odd += mos
    nbtts_odd += mos
    above_odd += mos
    below_odd += mos
    
    #Calculate the return from all the bets according to the odds
    for index, row in df.iterrows():
        #Reset the return value in each iteration
        total_profit = 0.0
        result_profit = 0.0
        btts_profit = 0.0
        above_profit = 0.0
        
        #Checking the win prediction
        if row['result_preds'] == 'X' and row['x_odds'] >= win_odd:
            result_bet_num += 1
            #When the prediction hit
            if row['result_actual'] == 'X':
                result_profit = bet_amount * row['x_odds'] - bet_amount
                
            #When the prediction did not hit
            else:
                result_profit = 0.0 - bet_amount
        else:
            pass

        #Checking the draw prediction
        if row['result_preds'] == 'D' and row['d_odds'] >= draw_odd:
            result_bet_num += 1
            #When the prediction hit
            if row['result_actual'] == 'D':
                result_profit = bet_amount * row['d_odds'] - bet_amount
                
            #When the prediction did not hit
            else:
                result_profit = 0.0 - bet_amount
        else:
            pass
            
        #Checking the btts prediction
        if row['btts_preds'] == 'Y' and row['btts_odds'] >= btts_odd:
            btts_bet_num += 1
            #When the prediction hit
            if row['btts_actual'] == 'Y':
                btts_profit = bet_amount * row['btts_odds'] - bet_amount
                
            #When the prediction did not hit
            else:
                btts_profit = 0.0 - bet_amount
        else:
            pass
        
        #Checking the nbtts prediction
        if row['btts_preds'] == 'N' and row['nbtts_odds'] >= nbtts_odd:
            btts_bet_num += 1
            #When the prediction hit
            if row['btts_actual'] == 'N':
                btts_profit = bet_amount * row['nbtts_odds'] - bet_amount
                
            #When the prediction did not hit
            else:
                btts_profit = 0.0 - bet_amount
        else:
            pass
        
        #Checking the above 2.5 prediction
        if row['above_preds'] == 'Y' and row['above_odds'] >= above_odd:
            above_bet_num += 1
            #When the prediction hit
            if row['above_actual'] == 'Y':
                above_profit = bet_amount * row['above_odds'] - bet_amount
                
            #When the prediction did not hit
            else:
                above_profit = 0.0 - bet_amount
        else:
            pass        

        #Checking the below 2.5 prediction
        if row['above_preds'] == 'N' and row['below_odds'] >= below_odd:
            above_bet_num += 1
            #When the prediction hit
            if row['above_actual'] == 'N':
                above_profit = bet_amount * row['below_odds'] - bet_amount
                
            #When the prediction did not hit
            else:
                above_profit = 0.0 - bet_amount
        else:
            pass
        
        #Add the profits to the list
        total_profit = result_profit + btts_profit + above_profit
        total_expected_profits.append(result_profit)
        total_expected_profits.append(btts_profit)
        total_expected_profits.append(above_profit)
        result_expected_profits.append(result_profit)
        btts_expected_profits.append(btts_profit)
        above_expected_profits.append(above_profit)

    #Calculate the results for each betting type
    total_bet_num = result_bet_num + btts_bet_num + above_bet_num
    
    #Create an array of just pure profits
    total_expected_profits = np.array(total_expected_profits)
    total_expected_profits = total_expected_profits[total_expected_profits != 0]
    
    result_expected_profits = np.array(result_expected_profits)
    result_expected_profits = result_expected_profits[result_expected_profits != 0]
    
    btts_expected_profits = np.array(btts_expected_profits)
    btts_expected_profits = btts_expected_profits[btts_expected_profits != 0]
    
    above_expected_profits = np.array(above_expected_profits)
    above_expected_profits = above_expected_profits[above_expected_profits != 0]    
    
    total = {
        'Number' : total_bet_num,
        'Total' : total_bet_num * bet_amount,
        'Return' : round((total_bet_num * bet_amount + sum(total_expected_profits)), 2),
        'Positive Returns' : np.sum(np.array(total_expected_profits) > 0, axis = 0),
        'Negative Returns' : np.sum(np.array(total_expected_profits) < 0, axis = 0),
        'Profit' : round((sum(total_expected_profits)),2),
        'Margin' : round((sum(total_expected_profits))/(total_bet_num * bet_amount),2),
        'Min Profit' : round(min(total_expected_profits),2),
        'Max Profit' : round(max(total_expected_profits),2),
        'Mean Profit' : round(np.mean(total_expected_profits),2),
        'Median Profit' : round(np.median(total_expected_profits),2),
        'Standard Dev' : round(np.std(total_expected_profits),2)            
    }

    result = {
        'Number' : result_bet_num,
        'Total' : result_bet_num * bet_amount,
        'Return' : round((result_bet_num * bet_amount + sum(result_expected_profits)), 2),
        'Positive Returns' : np.sum(np.array(result_expected_profits) > 0, axis = 0),
        'Negative Returns' : np.sum(np.array(result_expected_profits) < 0, axis = 0),
        'Profit' : round((sum(result_expected_profits)),2),
        'Margin' : round((sum(result_expected_profits))/(result_bet_num * bet_amount),2),
        'Min Profit' : round(min(result_expected_profits),2),
        'Max Profit' : round(max(result_expected_profits),2),
        'Mean Profit' : round(np.mean(result_expected_profits),2),
        'Median Profit' : round(np.median(result_expected_profits),2),
        'Standard Dev' : round(np.std(result_expected_profits),2)            
    }

    btts = {
        'Number' : btts_bet_num,
        'Total' : btts_bet_num * bet_amount,
        'Return' : round((btts_bet_num * bet_amount + sum(btts_expected_profits)), 2),
        'Positive Returns' : np.sum(np.array(btts_expected_profits) > 0, axis = 0),
        'Negative Returns' : np.sum(np.array(btts_expected_profits) < 0, axis = 0),
        'Profit' : round((sum(btts_expected_profits)),2),
        'Margin' : round((sum(btts_expected_profits))/(btts_bet_num * bet_amount),2),
        'Min Profit' : round(min(btts_expected_profits),2),
        'Max Profit' : round(max(btts_expected_profits),2),
        'Mean Profit' : round(np.mean(btts_expected_profits),2),
        'Median Profit' : round(np.median(btts_expected_profits),2),
        'Standard Dev' : round(np.std(btts_expected_profits),2)            
    }

    above = {
        'Number' : above_bet_num,
        'Total' : above_bet_num * bet_amount,
        'Return' : round((above_bet_num * bet_amount + sum(above_expected_profits)), 2),
        'Positive Returns' : np.sum(np.array(above_expected_profits) > 0, axis = 0),
        'Negative Returns' : np.sum(np.array(above_expected_profits) < 0, axis = 0),
        'Profit' : round((sum(above_expected_profits)),2),
        'Margin' : round((sum(above_expected_profits))/(above_bet_num * bet_amount),2),
        'Min Profit' : round(min(above_expected_profits),2),
        'Max Profit' : round(max(above_expected_profits),2),
        'Mean Profit' : round(np.mean(above_expected_profits),2),
        'Median Profit' : round(np.median(above_expected_profits),2),
        'Standard Dev' : round(np.std(above_expected_profits),2)            
    }
        
    return total, result, btts, above