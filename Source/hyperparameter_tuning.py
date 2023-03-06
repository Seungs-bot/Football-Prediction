import time
import optuna
from functions import save_artefact, load_artefact
from classes import Optuna_Objective

def hyperparameter_tuning(to_predict):
    model_filepath = "C:/Users/Lee Seung Soo/OneDrive/Desktop/Football Project/Source/models/"
    start_time = time.time()
    print(f"STARTING {to_predict.upper()} MODEL HYPERPARAMETER TUNING...")

    #Load the football model class and all the necessary values
    fm = load_artefact(f"{model_filepath}{to_predict}_initial_model.pickle")
    data = fm.get_data()
    train = fm.get_train_data()
    test = fm.get_test_data()
    predictors = fm.get_predictors()
    to_predict = fm.get_to_predict()
    mapping = fm.get_mapping()

    #Tuning Random Forest Model
    print("Beginning Tuning Random Forest Classification Model...")
    optuna_tuner = Optuna_Objective(model = 'rf', data = data, train = train, test = test, 
                                    predictors = predictors, to_predict = to_predict, mapping = mapping)
    study = optuna.create_study(direction = "maximize")
    study.optimize(optuna_tuner, n_trials = 100)
    print("Tuning Completed!")

    #Get, print and save the best trial
    best_trial, best_model, best_score, best_merged = optuna_tuner.get_best_trial()
    model_data = {
        'name' : 'rf',
        'model' : best_model,
        'preds' : best_merged
    }
    print("")
    print("--- Best Trial ---")
    print(f"Trial No: {best_trial}")
    print(f"Trial Score: {round(best_score, 3)}")
    print("-------------------")
    fm.add_tuned_model(model_data, 'rf')

    time.sleep(3)

    #Tuning Logistic Regression Model
    print("Beginning Tuning Logistic Regression Model...")
    optuna_tuner = Optuna_Objective(model = 'logreg', data = data, train = train, test = test, 
                                    predictors = predictors, to_predict = to_predict, mapping = mapping)
    study = optuna.create_study(direction = "maximize")
    study.optimize(optuna_tuner, n_trials = 100)
    print("Tuning Completed!")

    #Get and print out the best trial
    best_trial, best_model, best_score, best_merged = optuna_tuner.get_best_trial()
    model_data = {
        'name' : 'logreg',
        'model' : best_model,
        'preds' : best_merged
    }
    print("")
    print("--- Best Trial ---")
    print(f"Trial No: {best_trial}")
    print(f"Trial Score: {round(best_score, 3)}")
    print("-------------------")
    fm.add_tuned_model(model_data, 'logreg')

    time.sleep(3)

    #Tuning XGBClassifier Model
    print("Beginning Tuning XGBoost Classifier Model...")
    optuna_tuner = Optuna_Objective(model = 'xgb', data = data, train = train, test = test, 
                                    predictors = predictors, to_predict = to_predict, mapping = mapping)
    study = optuna.create_study(direction = "maximize")
    study.optimize(optuna_tuner, n_trials = 100)
    print("Tuning Completed!")

    #Get and print out the best trial
    best_trial, best_model, best_score, best_merged = optuna_tuner.get_best_trial()
    model_data = {
        'name' : 'xgb',
        'model' : best_model,
        'preds' : best_merged
    }
    print("--- Best Trial ---")
    print(f"Trial No: {best_trial}")
    print(f"Trial Score: {round(best_score, 3)}")
    print("-------------------")
    fm.add_tuned_model(model_data, 'xgb')    

    #Save the tuned football model class
    output_model = f"{model_filepath}{to_predict}_tuned_model.pickle"
    save_artefact(output_model, fm)

    end_time = time.time()
    print(f"FINISHED {to_predict.upper()} MODEL HYPERPARAMETER TUNING: {end_time - start_time} SECONDS")
    return

#hyperparameter_tuning("btts")