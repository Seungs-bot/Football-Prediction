import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

class MissingDict(dict):
    __missing__ = lambda self, key:key

class Football_Model:
    def __init__(self, data, predictors, to_predict, mapping, verbose=False, initial_rf_model=None, initial_logreg_model=None, initial_xgb_model=None,
                    tuned_rf_model=None, tuned_logreg_model=None, tuned_xgb_model=None, best_model=None, train=None, test=None):
        self.data = data
        self.predictors = predictors
        self.to_predict = to_predict
        self.mapping = mapping
        self.verbose = verbose
        self.initial_rf_model = initial_rf_model
        self.initial_logreg_model = initial_logreg_model
        self.initial_xgb_model = initial_xgb_model
        self.tuned_rf_model = tuned_rf_model
        self.tuned_logreg_model = tuned_logreg_model
        self.tuned_xgb_model = tuned_xgb_model
        self.best_model = best_model
        self.train = train
        self.test = test

    def get_data(self):
        return self.data
    
    def get_predictors(self):
        return self.predictors

    def get_to_predict(self):
        return self.to_predict

    def get_mapping(self):
        return self.mapping
    
    def get_initial_rf_model(self):
        return self.initial_rf_model

    def get_initial_logreg_model(self):
        return self.initial_logreg_model
    
    def get_initial_xgb_model(self):
        return self.initial_xgb_model

    def get_tuned_rf_model(self):
        return self.tuned_rf_model

    def get_tuned_logreg_model(self):
        return self.tuned_logreg_model

    def get_tuned_xgb_model(self):
        return self.tuned_xgb_model
    
    def get_train_data(self):
        return self.train

    def get_test_data(self):
        return self.test

    def train_initial_model(self):
        train = self.data[self.data["season"] != 2022]
        test = self.data[self.data["season"] == 2022]
        self.train, self.test = train, test
        print(f"Splitting data into {round(len(train)/(len(train)+len(test)),2)}:{round(len(test)/(len(train)+len(test)),2)} ratio")
        
        #Random Forest Classifier Model
        rf = RandomForestClassifier(n_estimators=150, min_samples_split=10, random_state=1)
        rf.fit(train[self.predictors], train[self.to_predict])
        preds = rf.predict(test[self.predictors])
        combined_rf = pd.DataFrame(dict(actual=test[self.to_predict], predicted = preds), index=test.index)
        combined_rf = combined_rf.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        precision_rf = precision_score(test[self.to_predict],preds)
        accuracy_rf = accuracy_score(test[self.to_predict], preds)
        model_rf = {
            'combined' : combined_rf,
            'accuracy' : accuracy_rf,
            'precision' : precision_rf,
            'model' : rf
        }
        self.initial_rf_model = model_rf
        
        #Logistic Regression Model
        logreg = LogisticRegression(random_state=1, max_iter = 500)
        logreg.fit(train[self.predictors], train[self.to_predict])
        preds = logreg.predict(test[self.predictors])
        combined_logreg = pd.DataFrame(dict(actual=test[self.to_predict], predicted = preds), index=test.index)
        combined_logreg = combined_logreg.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        precision_logreg = precision_score(test[self.to_predict],preds)
        accuracy_logreg = accuracy_score(test[self.to_predict], preds)
        model_logreg = {
            'combined' : combined_logreg,
            'accuracy' : accuracy_logreg,
            'precision' : precision_logreg,
            'model' : logreg
        }
        self.initial_logreg_model = model_logreg
        
        #XGBClassifier
        xgb = XGBClassifier(scale_pos_weight=1, learning_rate=0.005,
                        colsample_bytree=0.5, subsample=0.8, objective='binary:logistic',
                        n_estimators=1000, reg_alpha=0.2, max_depth=5, gamma=5, seed=82)
        xgb.fit(train[self.predictors], train[self.to_predict])
        preds = xgb.predict(test[self.predictors])
        combined_xgb = pd.DataFrame(dict(actual=test[self.to_predict], predicted = preds), index=test.index)
        combined_xgb = combined_xgb.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        precision_xgb = precision_score(test[self.to_predict],preds)
        accuracy_xgb = accuracy_score(test[self.to_predict], preds)
        model_xgb = {
            'combined' : combined_xgb,
            'accuracy' : accuracy_xgb,
            'precision' : precision_xgb,
            'model' : xgb
        }
        self.initial_xgb_model = model_xgb

    def evaluate_initial_model(self):
        print("")
        print("EVALUATING INITIAL MODELS...")
        
        print("")
        print("--- Random Forest Classifier ---")
        print(f"Accuracy: {self.initial_rf_model['accuracy']}")
        print(f"Precision: {self.initial_rf_model['precision']}")
        print("--------------------------------")

        print("")
        print("----- Logistic Regression -----")
        print(f"Accuracy: {self.initial_logreg_model['accuracy']}")
        print(f"Precision: {self.initial_logreg_model['precision']}")
        print("-------------------------------")

        print("")
        print("----- XGBoost Classifier -----")
        print(f"Accuracy: {self.initial_xgb_model['accuracy']}")
        print(f"Precision: {self.initial_xgb_model['precision']}")
        print("-------------------------------")

    def evaluate_initial_model_combined_objective(self):
        rf_pred_combined = self.initial_rf_model['combined']
        rf_pred_combined['new_team'] = rf_pred_combined['team'].map(self.mapping)
        rf_pred_combined['new_opponent'] = rf_pred_combined['opponent'].map(self.mapping)
        merged = rf_pred_combined.merge(rf_pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])
        print("")
        print("EVALAUTING INITIAL MODELS ON COMBINED OBJECTIVE...")

        print("")
        print("------ Random Forest Classifier ------")
        if self.to_predict == 'target':
            win_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]
            precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)
            print(f"Combined Objective Win/Lose: {round(precision_win, 5)}")

            draw_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_draw = len(draw_count[draw_count['result_x'] == 'D']) / len(draw_count)
            print(f"Combined Objective Draw: {round(precision_draw, 5)}")

        if self.to_predict == 'btts':
            btts_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)
            print(f"Combined Objective BTTS: {round(precision_btts, 5)}")

        if self.to_predict == 'above_2.5':        
            above_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)
            print(f"Combined Objective Above 2.5: {round(precision_above, 5)}")

            below_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_below = len(below_count[below_count["actual_x"] == 0]) / len(below_count)
            print(f"Combined Objective Below 2.5: {round(precision_below, 5)}")
        print("--------------------------------------")

        logreg_pred_combined = self.initial_logreg_model['combined']
        logreg_pred_combined['new_team'] = logreg_pred_combined['team'].map(self.mapping)
        logreg_pred_combined['new_opponent'] = logreg_pred_combined['opponent'].map(self.mapping)
        merged = logreg_pred_combined.merge(logreg_pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])
        
        print("")
        print("--------- Logistic Regression --------")
        if self.to_predict == 'target':
            win_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]
            precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)
            print(f"Combined Objective Win/Lose: {round(precision_win, 5)}")

            draw_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_draw = len(draw_count[draw_count['result_x'] == 'D']) / len(draw_count)
            print(f"Combined Objective Draw: {round(precision_draw, 5)}")

        if self.to_predict == 'btts':
            btts_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)
            print(f"Combined Objective BTTS: {round(precision_btts, 5)}")

        if self.to_predict == 'above_2.5':        
            above_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)
            print(f"Combined Objective Above 2.5: {round(precision_above, 5)}")

            below_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_below = len(below_count[below_count["actual_x"] == 0]) / len(below_count)
            print(f"Combined Objective Below 2.5: {round(precision_below, 5)}")
        print("--------------------------------------")

        xgb_pred_combined = self.initial_xgb_model['combined']
        xgb_pred_combined['new_team'] = xgb_pred_combined['team'].map(self.mapping)
        xgb_pred_combined['new_opponent'] = xgb_pred_combined['opponent'].map(self.mapping)
        merged = xgb_pred_combined.merge(xgb_pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])
        
        print("")
        print("--------- XGBoost Classifier --------")
        if self.to_predict == 'target':
            win_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]
            precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)
            print(f"Combined Objective Win/Lose: {round(precision_win, 5)}")

            draw_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_draw = len(draw_count[draw_count['result_x'] == 'D']) / len(draw_count)
            print(f"Combined Objective Draw: {round(precision_draw, 5)}")

        if self.to_predict == 'btts':
            btts_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)
            print(f"Combined Objective BTTS: {round(precision_btts, 5)}")

        if self.to_predict == 'above_2.5':        
            above_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)
            print(f"Combined Objective Above 2.5: {round(precision_above, 5)}")

            below_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_below = len(below_count[below_count["actual_x"] == 0]) / len(below_count)
            print(f"Combined Objective Below 2.5: {round(precision_below, 5)}")
        print("--------------------------------------")
        print("")

    def evalaute_tuned_model_combined_objective(self):

        model = self.tuned_rf_model['model']
        test = self.test
        preds = model.predict(test[self.predictors])
        pred_combined = pd.DataFrame(dict(actual=test[self.to_predict], predicted = preds), index=test.index)
        pred_combined = pred_combined.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        pred_combined['new_team'] = pred_combined['team'].map(self.mapping)
        pred_combined['new_opponent'] = pred_combined['opponent'].map(self.mapping)
        merged = pred_combined.merge(pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])

        print("")
        print("------ Random Forest Classifier ------")
        if self.to_predict == 'target':
            win_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]
            precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)
            print(f"Total Number of Cases: {len(win_count)}")
            print(f"Combined Objective Win/Lose: {round(precision_win, 5)}")

            draw_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_draw = len(draw_count[draw_count['result_x'] == 'D']) / len(draw_count)
            print(f"Total Number of Cases: {len(draw_count)}")
            print(f"Combined Objective Draw: {round(precision_draw, 5)}")

        if self.to_predict == 'btts':
            btts_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)
            print(f"Total Number of Cases: {len(btts_count)}")
            print(f"Combined Objective BTTS: {round(precision_btts, 5)}")

        if self.to_predict == 'above_2.5':        
            above_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)
            print(f"Total Number of Cases: {len(above_count)}")
            print(f"Combined Objective Above 2.5: {round(precision_above, 5)}")

            below_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_below = len(below_count[below_count["actual_x"] == 0]) / len(below_count)
            print(f"Total Number of Cases: {len(below_count)}")
            print(f"Combined Objective Below 2.5: {round(precision_below, 5)}")
        print("--------------------------------------")
        
        model = self.tuned_logreg_model['model']
        test = self.test
        preds = model.predict(test[self.predictors])
        pred_combined = pd.DataFrame(dict(actual=test[self.to_predict], predicted = preds), index=test.index)
        pred_combined = pred_combined.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        pred_combined['new_team'] = pred_combined['team'].map(self.mapping)
        pred_combined['new_opponent'] = pred_combined['opponent'].map(self.mapping)
        merged = pred_combined.merge(pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])

        print("")
        print("--------- Logistic Regression --------")
        if self.to_predict == 'target':
            win_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]
            precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)
            print(f"Total Number of Cases: {len(win_count)}")
            print(f"Combined Objective Win/Lose: {round(precision_win, 5)}")

            draw_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_draw = len(draw_count[draw_count['result_x'] == 'D']) / len(draw_count)
            print(f"Total Number of Cases: {len(draw_count)}")
            print(f"Combined Objective Draw: {round(precision_draw, 5)}")

        if self.to_predict == 'btts':
            btts_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)
            print(f"Total Number of Cases: {len(btts_count)}")
            print(f"Combined Objective BTTS: {round(precision_btts, 5)}")

        if self.to_predict == 'above_2.5':        
            above_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)
            print(f"Total Number of Cases: {len(above_count)}")
            print(f"Combined Objective Above 2.5: {round(precision_above, 5)}")

            below_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_below = len(below_count[below_count["actual_x"] == 0]) / len(below_count)
            print(f"Total Number of Cases: {len(below_count)}")
            print(f"Combined Objective Below 2.5: {round(precision_below, 5)}")
        print("--------------------------------------")

        model = self.tuned_xgb_model['model']
        test = self.test
        preds = model.predict(test[self.predictors])
        pred_combined = pd.DataFrame(dict(actual=test[self.to_predict], predicted = preds), index=test.index)
        pred_combined = pred_combined.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        pred_combined['new_team'] = pred_combined['team'].map(self.mapping)
        pred_combined['new_opponent'] = pred_combined['opponent'].map(self.mapping)
        merged = pred_combined.merge(pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])
        
        print("")
        print("--------- XGBoost Classifier --------")
        if self.to_predict == 'target':
            win_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]
            precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)
            print(f"Total Number of Cases: {len(win_count)}")
            print(f"Combined Objective Win/Lose: {round(precision_win, 5)}")

            draw_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_draw = len(draw_count[draw_count['result_x'] == 'D']) / len(draw_count)
            print(f"Total Number of Cases: {len(draw_count)}")
            print(f"Combined Objective Draw: {round(precision_draw, 5)}")

        if self.to_predict == 'btts':
            btts_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)
            print(f"Total Number of Cases: {len(btts_count)}")
            print(f"Combined Objective BTTS: {round(precision_btts, 5)}")

        if self.to_predict == 'above_2.5':        
            above_count = merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 1)]
            precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)
            print(f"Total Number of Cases: {len(above_count)}")
            print(f"Combined Objective Above 2.5: {round(precision_above, 5)}")

            below_count = merged[(merged["predicted_x"] == 0) & (merged["predicted_y"] == 0)]
            precision_below = len(below_count[below_count["actual_x"] == 0]) / len(below_count)
            print(f"Total Number of Cases: {len(below_count)}")
            print(f"Combined Objective Below 2.5: {round(precision_below, 5)}")
        print("--------------------------------------")
        print("")
    
    def get_initial_model(self, model):
        if model == "rf":
            return self.initial_rf_model
        if model == 'logreg':
            return self.initial_logreg_model
        if model == 'xgb':
            return self.initial_xgb_model

    def add_tuned_model(self, model_data, model_name):
        if model_name == 'rf':
            self.tuned_rf_model = model_data
        if model_name == 'logreg':
            self.tuned_logreg_model = model_data
        if model_name == 'xgb':
            self.tuned_xgb_model = model_data

    def update_data(self, data):
        self.data = data

    def get_predictions(self, data):

        self.test = data

        #Get predictions for the Random Forest Model
        model_rf = self.tuned_rf_model['model']
        preds = model_rf.predict(self.test[self.predictors])
        pred_combined = pd.DataFrame(dict(actual=self.test[self.to_predict], predicted = preds), index=self.test.index)
        pred_combined = pred_combined.merge(self.data[["date", "team", "opponent", "result", "round"]], left_index=True, right_index=True)
        pred_combined['new_team'] = pred_combined['team'].map(self.mapping)
        pred_combined['new_opponent'] = pred_combined['opponent'].map(self.mapping)
        merged_rf = pred_combined.merge(pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])

        #Get predictions for the Logistic Regression Model
        model_logreg = self.tuned_logreg_model['model']
        preds = model_logreg.predict(self.test[self.predictors])
        pred_combined = pd.DataFrame(dict(actual=self.test[self.to_predict], predicted = preds), index=self.test.index)
        pred_combined = pred_combined.merge(self.data[["date", "team", "opponent", "result", "round"]], left_index=True, right_index=True)
        pred_combined['new_team'] = pred_combined['team'].map(self.mapping)
        pred_combined['new_opponent'] = pred_combined['opponent'].map(self.mapping)
        merged_logreg = pred_combined.merge(pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])

        #Get predictions for the XGB Model
        model_xgb = self.tuned_xgb_model['model']
        preds = model_xgb.predict(self.test[self.predictors])
        pred_combined = pd.DataFrame(dict(actual=self.test[self.to_predict], predicted = preds), index=self.test.index)
        pred_combined = pred_combined.merge(self.data[["date", "team", "opponent", "result", "round"]], left_index=True, right_index=True)
        pred_combined['new_team'] = pred_combined['team'].map(self.mapping)
        pred_combined['new_opponent'] = pred_combined['opponent'].map(self.mapping)
        merged_xgb = pred_combined.merge(pred_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])

        return merged_rf, merged_logreg, merged_xgb

class Optuna_Objective:
    def __init__(self, model, data, train, test, predictors, to_predict, mapping, trial_results = None):
        self.model = model
        self.data = data
        self.train = train
        self.test = test
        self.predictors = predictors
        self.to_predict = to_predict
        self.mapping = mapping
        self.trial_results = {
            'models' : [], 
            'scores' : [],
            'merged' : []
        }

    def __call__(self, trial):
        if self.model == 'rf':
            #Set the parameters for Random Forest Clasifier
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            criterion = trial.suggest_categorical("criterion", ["gini", "log_loss"])
            max_depth = trial.suggest_int("max_depth", 50, 200)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 1000, 5000)
            min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.1)
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            if bootstrap == False:
                oob_score = False
            else:
                oob_score = trial.suggest_categorical("oob_score", [True, False])
            random_state = trial.suggest_int("random_state", 0, 50)
            warm_start = trial.suggest_categorical("warm_start", [True, False])
            ccp_alpha = trial.suggest_loguniform("ccp_alpha", 0.01, 0.02)
            if bootstrap == False:
                max_samples = None
            else:
                max_samples = trial.suggest_int("max_samples", 1, 50)

            #Fit the model according to suggested parameters
            model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split,
                                            min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease,
                                            bootstrap = bootstrap, oob_score = oob_score, random_state = random_state, warm_start = warm_start, ccp_alpha = ccp_alpha, max_samples = max_samples)
            model.fit(self.train[self.predictors], self.train[self.to_predict])
            
            #Find the objective metric for both the train and test set
            test_merged, train_merged = self.find_predictions(model = model)
            
            #Obtimize the average of train and test set objective scores to prevent overfitting and underfitting
            objective = self.calculate_objective(test_merged = test_merged, train_merged = train_merged)
            
            #Add the trial result into the list
            self.trial_results['models'].append(model)
            self.trial_results['scores'].append(objective)
            self.trial_results['merged'].append(test_merged)
            return objective
        
        if self.model == 'logreg':
            #Set the parameters for logistic regression
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", 'none'])
            param_C = trial.suggest_float("C", 0.0001, 1.0)
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            
            if penalty == 'l2':
                solver = trial.suggest_categorical("solver1", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
            elif penalty == 'none':
                solver = trial.suggest_categorical("solver2", ['lbfgs', 'newton-cg','sag', 'saga'])
            elif penalty == 'elasticnet':
                solver = trial.suggest_categorical("solver3", ['saga','saga'])
            else:
                solver = trial.suggest_categorical("solver4", ['liblinear', 'saga'])
            
            if solver == 'liblinear' and fit_intercept == True:
                intercept_scaling = trial.suggest_float("intercept_scaling1", 0.1, 5.0)
            else:
                intercept_scaling = 1
            
            random_state = trial.suggest_int("random_state", 0, 50)
            max_iter = trial.suggest_int("max_iter", 50, 2000)
            warm_start = trial.suggest_categorical("warm_start", [True, False])

            if penalty == 'elasticnet':
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            else:
                l1_ratio = None

            #Fit the model according to suggested parameters
            model = LogisticRegression(penalty = penalty, C = param_C, fit_intercept = fit_intercept, solver = solver,
                                        intercept_scaling = intercept_scaling, random_state = random_state,
                                        max_iter = max_iter, warm_start = warm_start, l1_ratio = l1_ratio)
            model.fit(self.train[self.predictors], self.train[self.to_predict])

            #Find the objective metric for both the train and test set
            test_merged, train_merged = self.find_predictions(model = model)

            #Optimize the average of train and test set obejctive scores to prevent overfitting and underfitting
            objective = self.calculate_objective(test_merged = test_merged, train_merged = train_merged)
            
            #Add the trial result into the list
            self.trial_results['models'].append(model)
            self.trial_results['scores'].append(objective)
            self.trial_results['merged'].append(test_merged)
            return objective

        if self.model == 'xgb':
            #Set the parameters for XGBoost classifier
            booster = trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart'])
            
            #When the tree booster is chosen
            if booster == 'gbtree':
                objective = trial.suggest_categorical("objective", ['binary:logistic', 'binary:logitraw', 'binary:hinge'])
                eta = trial.suggest_float("eta", 0, 1)
                gamma = trial.suggest_float("gamma", 0, 5)
                max_depth = trial.suggest_int("max_depth", 6, 20)
                min_child_weight = trial.suggest_float("min_child_weight", 1, 5)
                max_delta_step = trial.suggest_int("max_delta_step", 0, 10)
                subsample = trial.suggest_float("subsample", 0.1, 1)
                reg_lambda = trial.suggest_float("reg_lambda", 0.1, 3)
                reg_alpha = trial.suggest_float("reg_alpha", 0.1, 3)
                tree_method = trial.suggest_categorical("tree_method", ['approx', 'hist'])
                grow_policy = trial.suggest_categorical("grow_policy", ['depthwise', 'lossguide'])
                max_leaves = trial.suggest_int("max_leaves", 0, 10)
                max_bin = trial.suggest_int("max_bin", 256, 512)

                #Fit the model according to suggested parameters
                model = XGBClassifier(objective = objective, 
                                    booster = booster,
                                    eta = eta,
                                    gamma = gamma,
                                    max_depth = max_depth,
                                    min_child_weight = min_child_weight,
                                    max_delta_step = max_delta_step,
                                    subsample = subsample,
                                    reg_lambda = reg_lambda,
                                    reg_alpha = reg_alpha,
                                    tree_method = tree_method,
                                    grow_policy = grow_policy,
                                    max_leaves = max_leaves,
                                    max_bin = max_bin)
                model.fit(self.train[self.predictors], self.train[self.to_predict])

                #Find the objective metric for both the train and test set
                test_merged, train_merged = self.find_predictions(model = model)

                #Optimize the average of train and test set obejctive scores to prevent overfitting and underfitting
                objective = self.calculate_objective(test_merged = test_merged, train_merged = train_merged)
                
                #Add the trial result into the list
                self.trial_results['models'].append(model)
                self.trial_results['scores'].append(objective)
                self.trial_results['merged'].append(test_merged)
                return objective

            #When the dart booster is chosen
            elif booster == 'dart':
                objective = trial.suggest_categorical("objective", ['binary:logistic', 'binary:logitraw', 'binary:hinge'])
                sample_type = trial.suggest_categorical("sample_type", ['uniform', 'weighted'])
                normalize_type = trial.suggest_categorical("normalize_type", ['tree', 'forest'])
                rate_drop = trial.suggest_float("rate_drop", 0.0, 1.0)
                one_drop = trial.suggest_categorical("one_drop", [1, 0])
                skip_drop = trial.suggest_float("skip_drop", 0.0, 1.0)

                #Fit the model according to suggested parameters
                model = XGBClassifier(objective = objective,
                                        booster = booster,
                                        sample_type = sample_type,
                                        normalize_type = normalize_type,
                                        rate_drop = rate_drop,
                                        one_drop = one_drop,
                                        skip_drop = skip_drop)
                model.fit(self.train[self.predictors], self.train[self.to_predict])

                #Find the objective metric for both the train and test set
                test_merged, train_merged = self.find_predictions(model = model)

                #Optimize the average of train and test set objective scores to prevent overfitting and underfitting
                objective = self.calculate_objective(test_merged = test_merged, train_merged = train_merged)
                
                #Add the trial result into the list
                self.trial_results['models'].append(model)
                self.trial_results['scores'].append(objective)
                self.trial_results['merged'].append(test_merged)
                return objective

            #When the linear booster is chosen
            else:
                objective = trial.suggest_categorical("objective", ['binary:logistic', 'binary:logitraw', 'binary:hinge'])
                reg_lambda = trial.suggest_float("reg_lambda", 0.1, 3)
                reg_alpha = trial.suggest_float("reg_alpha", 0.1, 3)
                updater = trial.suggest_categorical("updater", ['shotgun', 'coord_descent'])
                if updater == 'shotgun':
                    feature_selector = trial.suggest_categorical("feature_selector1", ['cyclic', 'shuffle'])
                else:
                    feature_selector = trial.suggest_categorical("feature_selector2", ['cyclic', 'shuffle', 'random', 'greedy', 'thrifty'])
                top_k = trial.suggest_int("top_k", 0,5)

                #Fit the model according to suggested parameters
                model = XGBClassifier(objective = objective,
                                        booster = booster,
                                        reg_lambda = reg_lambda,
                                        reg_alpha = reg_alpha,
                                        updater = updater,
                                        feature_selector = feature_selector,
                                        top_k = top_k)
                model.fit(self.train[self.predictors], self.train[self.to_predict])

                #Find the objective metric for both the train and test set
                test_merged, train_merged = self.find_predictions(model = model)

                #Optimize the average of train and test set objective scores to prevent overfitting and underfitting
                objective = self.calculate_objective(test_merged = test_merged, train_merged = train_merged)
                
                #Add the trial result into the list
                self.trial_results['models'].append(model)
                self.trial_results['scores'].append(objective)
                self.trial_results['merged'].append(test_merged)
                return objective
    
    def find_predictions(self, model):
        #Find the objective metric for both the train and test set
        test_preds = model.predict(self.test[self.predictors])
        test_combined = pd.DataFrame(dict(actual=self.test[self.to_predict], predicted = test_preds), index=self.test.index)
        test_combined = test_combined.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        test_combined['new_team'] = test_combined['team'].map(self.mapping)
        test_combined['new_opponent'] = test_combined['opponent'].map(self.mapping)
        test_merged = test_combined.merge(test_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])
        
        train_preds = model.predict(self.train[self.predictors])
        train_combined = pd.DataFrame(dict(actual=self.train[self.to_predict], predicted = train_preds), index=self.train.index)
        train_combined = train_combined.merge(self.data[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        train_combined['new_team'] = train_combined['team'].map(self.mapping)
        train_combined['new_opponent'] = train_combined['opponent'].map(self.mapping)
        train_merged = train_combined.merge(train_combined, left_on = ["date", "new_team"], right_on = ["date", "new_opponent"])

        return test_merged, train_merged

    def calculate_objective(self, test_merged, train_merged):
        #Finding the objective metric when the y-variable to predict == 'target'
        if self.to_predict == 'target':
            
            #Finding the precision of the test set
            win_count = test_merged[(test_merged["predicted_x"] == 1) & (test_merged["predicted_y"] == 0)]
            if len(win_count) == 0 or len(win_count) < 30:
                test_precision_win = 0
            else:
                test_precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)

            #Finding the precision of the train set
            win_count = train_merged[(train_merged["predicted_x"] == 1) & (train_merged["predicted_y"] == 0)]
            if len(win_count) == 0 or len(win_count) < 30:
                train_precision_win = 0
            else:
                train_precision_win = len(win_count[win_count['actual_x'] == 1]) / len(win_count)

            #Checking whether there has been an overfitting of the model
            #The model is deemed to have been overfitted if the difference in objective score between train and test set is more than 0.2
            if train_precision_win - test_precision_win >= 0.2:
                return 0
            else:
                return (train_precision_win + test_precision_win) / 2

        #Finding the objective metric when the y-variable to predict == 'btts'
        if self.to_predict == 'btts':
            
            #Finding the precision of the test set:
            btts_count = test_merged[(test_merged["predicted_x"] == 1) & (test_merged["predicted_y"] == 1)]
            if len(btts_count) == 0 or len(btts_count) < 30:
                test_precision_btts = 0
            else:
                test_precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)

            #Finding the precision of the train set:
            btts_count = train_merged[(train_merged["predicted_x"] == 1) & (train_merged["predicted_y"] == 1)]
            if len(btts_count) == 0 or len(btts_count) < 30:
                train_precision_btts = 0
            else:
                train_precision_btts = len(btts_count[(btts_count['actual_x'] == 1) & (btts_count['actual_y'] == 1)]) / len(btts_count)

            #Checking whether there has been an overfitting of the model
            #The model is deemed to have been overfitted if the difference in objective score between train and test set is more than 0.2
            if train_precision_btts - test_precision_btts >= 0.2:
                return 0
            else:
                return (test_precision_btts + train_precision_btts) / 2

        #Finding the objective metric when the y-variable to predict == 'above_2.5
        if self.to_predict == 'above_2.5':        
            
            #Finding the precision of the test set:
            above_count = test_merged[(test_merged["predicted_x"] == 1) & (test_merged["predicted_y"] == 1)]
            if len(above_count) == 0 or len(above_count) < 30:
                test_precision_above = 0
            else:
                test_precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)

            #Finding the precision of the train set:
            above_count = train_merged[(train_merged["predicted_x"] == 1) & (train_merged["predicted_y"] == 1)]
            if len(above_count) == 0 or len(above_count) < 30:
                train_precision_above = 0
            else:
                train_precision_above = len(above_count[above_count["actual_x"] == 1]) / len(above_count)

            #Checking whether there has been an overfitting of the model
            #The model is deemed to have been overfitted if the difference in objective score between train and test set is more than 0.2
            if train_precision_above - test_precision_above >= 0.2:
                return 0
            else:
                return (test_precision_above + train_precision_above) / 2

    def get_best_trial(self):
        scores_array = np.array(self.trial_results['scores'])
        models_array = np.array(self.trial_results['models'], dtype='object')
        merged_array = np.array(self.trial_results['merged'])

        #Get the index of the highest score value
        max_index = scores_array.argmax()

        return max_index, models_array[max_index], scores_array[max_index], merged_array[max_index]

