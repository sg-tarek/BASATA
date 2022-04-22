# Graphical libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("classic")
sns.set(style ='darkgrid')

# ML Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
import xgboost
from catboost import CatBoostClassifier
import lightgbm as lgb

# ML Regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from catboost import CatBoostRegressor

# Performance metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer 

# Hide warnings when running the script
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')

# Explainable AI
import lime
import lime.lime_tabular
import shap

# Other
import pandas as pd
import numpy as np
from tqdm import tqdm
import sweetviz as sv

def null(DataFrame):
    """
    The null function gives you insight to number of null values
    """
    print(DataFrame.isnull().sum())
    plt.figure(figsize=(10,6)) #Change size if you have many features
    sns.heatmap(DataFrame.isna(), cbar=False, cmap='binary', yticklabels=False)

def correlation(DataFrame):
    """
    The correlation function gives you insight to the correlation between numerical features
    """
    corr = DataFrame.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool)) #Masking the diagonoal and above

    plt.figure(figsize=(10,6))
    sns.heatmap(corr, cmap='RdYlGn', yticklabels=True, annot=True, fmt='.2f', mask=mask, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

def FID(DataFrame, ML_model, plot=True, length=5, height=5):
    """
    Function for creating a Feature Importance Dataframe (FID) and plotting it
    Only models with a feature_importances_ attribute are supported
    """
    column_names    = DataFrame.columns
    importances     = ML_model.feature_importances_
    
    df = pd.DataFrame({'feature': column_names,
    'feature_importance': importances}) \
        .sort_values('feature_importance', ascending = False) \
        .reset_index(drop = True)

    # Plot the Feature Importance Dataframe
    if plot == True:
        fig, ax = plt.subplots(figsize=(length, height))
        title = 'Feature Importance - Intrinsic method'
        sns.barplot(x = 'feature_importance', y = 'feature', data = df, orient = 'h', palette="rocket", saturation=.5).set_title(title, fontsize = 20)
    
    return df

def compare_models(X, y, random_seed=None, classification=True):
    """
    Function for comparing feature importance of different ML models
    Only models with a feature_importances_ attribute are supported
    """

    if classification==True:
        # Import Classification models
        rf = RandomForestClassifier(n_jobs=-1, random_state=random_seed)
        gbdt = GradientBoostingClassifier(random_state=random_seed)
        ab = AdaBoostClassifier(random_state=random_seed)
        dt = DecisionTreeClassifier(random_state=random_seed)
        xgb = xgboost.XGBClassifier(n_jobs=-1, verbosity = 0, use_label_encoder=False, random_state=random_seed)
        cb = CatBoostClassifier(random_state=random_seed)
        lgbm = lgb.LGBMClassifier(n_jobs=-1, random_state=random_seed)
    else:
        # Import Regression models
        rf = RandomForestRegressor(n_jobs=-1, random_state=random_seed)
        gbdt = GradientBoostingRegressor(random_state=random_seed)
        ab = AdaBoostRegressor(random_state=random_seed)
        dt = DecisionTreeRegressor(random_state=random_seed)
        xgb = xgboost.XGBRegressor(n_jobs=-1, verbosity = 0, random_state=random_seed)
        cb = CatBoostRegressor(random_state=random_seed)
        lgbm = lgb.LGBMRegressor(n_jobs=-1, random_state=random_seed)

    # Fit the models
    rf.fit(X, y)
    gbdt.fit(X, y)
    ab.fit(X, y)
    dt.fit(X, y)
    xgb.fit(X, y)
    cb.fit(X, y)
    lgbm.fit(X, y)

    # Add the results
    model_list = {
        'Random Forest': rf,
        'Gradient Boosting': gbdt,
        'AdaBoost': ab,
        'Decision Tree': dt,
        'XGBoost': xgb,
        'CatBoost': cb,
        'LightGBM': lgbm
    }

    # Create a dataframe to store the results of the models
    column_names    = X.columns
    features = {}
    
    for key, value in model_list.items():
        importances     = value.feature_importances_

        df = pd.DataFrame({'feature': column_names,
        'feature_importance': importances}) \
            .sort_values('feature_importance', ascending = False) \
            .reset_index(drop = True)

        features[key] = df.feature

    df = pd.DataFrame.from_dict(features, orient='index').T
            
    return df


def automl(X_test, X_train, y_train, y_test, random_seed=None, classification=True):
    """
    Function for evaluating classification and regressional models
    """

    y_pred_train = {} #Used in accuracy_score
    y_pred_test = {} #Used in accuracy_score

    y_pred_prob_train = {} #Used in roc_curve
    y_pred_prob_test = {} #Used in roc_curve

    if classification==True:
        # Import ML models
        rf = RandomForestClassifier(n_jobs=-1, random_state=random_seed)
        gbdt = GradientBoostingClassifier(random_state=random_seed)
        ab = AdaBoostClassifier(random_state=random_seed)
        dt = DecisionTreeClassifier(random_state=random_seed)
        knn = KNeighborsClassifier(n_jobs=-1)
        svm = SGDClassifier(n_jobs=-1, random_state=random_seed)
        mlp = MLPClassifier(random_state=random_seed )
        mlp.out_activation_ = 'logistic' #used for binary classification
        xgb = xgboost.XGBClassifier(n_jobs=-1, verbosity = 0, use_label_encoder=False, random_state=random_seed)
        cb = CatBoostClassifier(random_state=random_seed)
        lgbm = lgb.LGBMClassifier(n_jobs=-1, random_state=random_seed)
    else:
        # Import ML models
        rf = RandomForestRegressor(n_jobs=-1, random_state=random_seed)
        gbdt = GradientBoostingRegressor(random_state=random_seed)
        ab = AdaBoostRegressor(random_state=random_seed)
        dt = DecisionTreeRegressor(random_state=random_seed)
        knn = KNeighborsRegressor(n_jobs=-1)
        svm = SGDRegressor(n_jobs=-1, random_state=random_seed)
        mlp = MLPRegressor(random_state=random_seed)
        xgb = xgboost.XGBRegressor(n_jobs=-1, verbosity = 0, random_state=random_seed)
        cb = CatBoostRegressor(random_state=random_seed)
        lgbm = lgb.LGBMRegressor(n_jobs=-1, random_state=random_seed)

    # Fit the models
    rf.fit(X_train, y_train)
    gbdt.fit(X_train, y_train)
    ab.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    cb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    # Add the results
    model_list = {
        'Random Forest': rf,
        'Gradient Boosting': gbdt,
        'AdaBoost': ab,
        'Decision Tree': dt,
        'KNN': knn,
        'SVM': svm,
        'MLP': mlp,
        'XGBoost': xgb,
        'CatBoost': cb,
        'LightGBM': lgbm
    }

    for key, value in model_list.items():
        y_pred_train[key] = value.predict(X_train)
        y_pred_test[key] = value.predict(X_test)
        if key == 'SVM':
            y_pred_prob_train[key] = value.decision_function(X_train)
            y_pred_prob_test[key] = value.decision_function(X_test)
        else:
            y_pred_prob_train[key] = value.predict_proba(X_train)[:, 1]
            y_pred_prob_test[key] = value.predict_proba(X_test)[:, 1]

    # Evaluate the models
    if classification==True:
        accuracy = {}
        recall = {}
        precision = {}
        F1 = {}
        balanced_accuracy = {}
        roc_score = {}

        for algo in tqdm(model_list, desc='Evaluating models'):
            accuracy[algo] = accuracy_score(y_test, y_pred_test[algo])
            recall[algo] = recall_score(y_test, y_pred_test[algo])
            precision[algo] = precision_score(y_test, y_pred_test[algo])
            F1[algo] = f1_score(y_test, y_pred_test[algo])
            balanced_accuracy[algo] = balanced_accuracy_score(y_test, y_pred_test[algo])
            roc_score[algo] = roc_auc_score(y_test, y_pred_prob_test[algo])

        df_performance = pd.DataFrame([accuracy, recall, precision, F1, balanced_accuracy, roc_score])
        df_performance.columns = model_list
        df_performance.index = ["Accuracy", "Recall", "Precision", "F1", "Balanced Accuracy", "ROC AUC"]
        df_transposed = df_performance.T
        df_sorted = df_transposed.sort_values(by ='Accuracy', ascending=False)

        # Print ROC curves in the same plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for algo in tqdm(model_list, desc='ROC curves'):
            
            # fpr = False Positive Rate
            # tpr = True Positive Rate
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test[algo])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=algo + ' (AUC = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve')
        ax.legend(loc="lower right")
        plt.show()

        # Print Precision Recall Curve in the same plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, value in tqdm(model_list.items(), desc='Precision Recall Curve'):
            plot_precision_recall_curve(value, X_test, y_test, name = key, ax = plt.gca())
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Recall Curve')
        ax.legend(loc="lower left")
        plt.show()
    else:
        mean_absolute = {}
        mean_squared = {}
        r2 = {}
        explained_variance = {}

        for algo in tqdm(model_list, desc='Evaluating models'):
            mean_absolute[algo] = mean_absolute_error(y_test, y_pred_test[algo])
            mean_squared[algo] = mean_squared_error(y_test, y_pred_test[algo])
            r2[algo] = r2_score(y_test, y_pred_test[algo])
            explained_variance[algo] = explained_variance_score(y_test, y_pred_test[algo])

        df_performance = pd.DataFrame([mean_absolute, mean_squared, r2, explained_variance])
        df_performance.columns = model_list
        df_performance.index = ["Mean Absolute Error", "Mean Squared Error", "R2 Score", "Explained Variance Score"]    
        df_transposed = df_performance.T
        df_sorted = df_transposed.sort_values(by ='Mean Squared Error', ascending=False)

    return df_sorted.style.highlight_max(axis=0)

def tuning(X, y, model='rf', GridSearch=False, scoring=accuracy_score, random_seed=None, classification=True):
    """
    Function for hyperparameter tuning for models using GridSearchCV or RandomizedSearchCV (default)
    Model: 'rf', 'gbdt', 'ab', 'dt', 'knn', 'svm', 'mlp', 'xgb', 'cb', 'lgbm'
    scoring: accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
    """

    if classification==True:
        # Import ML models
        rf = RandomForestClassifier(n_jobs=-1, random_state=random_seed)
        gbdt = GradientBoostingClassifier(random_state=random_seed)
        ab = AdaBoostClassifier(random_state=random_seed)
        dt = DecisionTreeClassifier(random_state=random_seed)
        knn = KNeighborsClassifier(n_jobs=-1)
        svm = SGDClassifier(n_jobs=-1, random_state=random_seed)
        mlp = MLPClassifier(random_state=random_seed )
        mlp.out_activation_ = 'logistic' #used for binary classification
        xgb = xgboost.XGBClassifier(n_jobs=-1, verbosity = 0, use_label_encoder=False, random_state=random_seed)
        cb = CatBoostClassifier(random_state=random_seed)
        lgbm = lgb.LGBMClassifier(n_jobs=-1, random_state=random_seed)
    else:
        # Import ML models
        rf = RandomForestRegressor(n_jobs=-1, random_state=random_seed)
        gbdt = GradientBoostingRegressor(random_state=random_seed)
        ab = AdaBoostRegressor(random_state=random_seed)
        dt = DecisionTreeRegressor(random_state=random_seed)
        knn = KNeighborsRegressor(n_jobs=-1)
        svm = SGDRegressor(n_jobs=-1, random_state=random_seed)
        mlp = MLPRegressor(random_state=random_seed)
        xgb = xgboost.XGBRegressor(n_jobs=-1, verbosity = 0, random_state=random_seed)
        cb = CatBoostRegressor(random_state=random_seed)
        lgbm = lgb.LGBMRegressor(n_jobs=-1, random_state=random_seed)

    # The models to be tuned
    model_list = {
        'rf': rf,
        'gbdt': gbdt,
        'ab': ab,
        'dt': dt,
        'knn': knn,
        'svm': svm,
        'mlp': mlp,
        'xgb': xgb,
        'cb': cb,
        'lgbm': lgbm
    }

    # Hyperparameters to be tuned
    hyperparameters = {
        'rf': {
            'n_estimators': [10, 50, 100, 200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 10]
        },
        'gbdt': {
            'n_estimators': [10, 50, 100, 200, 500],
            'learning_rate': [0.1, 0.5, 1],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 10]
        },
        'ab': {
            'n_estimators': [10, 50, 100, 200, 500],
            'learning_rate': [0.1, 0.5, 1]
        },
        'dt': {
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 10]
        },
        'knn': {
            'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'svm': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['auto', 0.1, 0.5, 1]
        },
        'mlp': {
            'hidden_layer_sizes': [(100,), (50,), (25,), (10,), (5,)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.05, 0.1, 0.5, 1]
        },
        'xgb': {
            'n_estimators': [10, 50, 100, 200, 500],
            'learning_rate': [0.1, 0.5, 1],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        'cb': {
            'n_estimators': [10, 50, 100, 200, 500],
            'learning_rate': [0.1, 0.5, 1],
            'depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'l2_leaf_reg': [1, 5, 10, 50, 100, 500, 1000]
        },
        'lgbm': {
            'n_estimators': [10, 50, 100, 200, 500],
            'learning_rate': [0.1, 0.5, 1],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'num_leaves': [2, 5, 10, 20, 50, 100, 200, 500]
        }
    }

    # Tuning the model
    scorer = make_scorer(scoring)

    if GridSearch==True:
        # GridSearchCV
        grid_search = GridSearchCV(model_list[model], param_grid=hyperparameters[model], cv=5, scoring=scorer, n_jobs=-1)
        grid_result = grid_search.fit(X, y)

        print('The GridSearchCV found the following best parameters:')
        print('Best score for {}: {}'.format(model, grid_result.best_score_))
        print('Best estimator for {}: {}'.format(model, grid_result.best_estimator_))

    else:
        # RandomizedSearchCV
        grid_search = RandomizedSearchCV(model_list[model], param_distributions=hyperparameters[model], cv=5, scoring=scorer, n_jobs=-1)
        grid_result = grid_search.fit(X, y)

        print('The RandomizedSearchCV found the following best parameters:')
        print('Best score for {}: {}'.format(model, grid_result.best_score_))
        print('Best estimator for {}: {}'.format(model, grid_result.best_estimator_))
    
    return grid_result

def xai(X_train, X_test, model=None, mode='classification', observation=0, feature_selection='none'):
    """
    The eXplainable AI (XAI) function is used to explain the model's predictions. The model uses LIME to explain the predictions of the model.
    """

    if mode == 'classification':
        explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data = X_train.values,
        mode = mode,
        feature_names = X_train.columns,
        feature_selection = feature_selection #Set to 'auto' or 'lasso_path' in case of feature selection, I have set it to none to see all features
        )
    elif mode == 'regression':
        explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data = X_train.values,
        mode = mode,
        feature_names = X_train.columns,
        feature_selection = feature_selection #Set to 'auto' or 'lasso_path' in case of feature selection, I have set it to none to see all features
        )

    exp = explainer.explain_instance(
        data_row=X_test.iloc[observation], #Specific observation to predict
        predict_fn=model.predict_proba #The model GBDT in this case
    )
    exp.show_in_notebook(show_table=True)

    exp.as_pyplot_figure();

    return exp

def eda(DataFrame):
    """
    The Exploratory Data Analysis (EDA) function is used to explore the data.
    Using the sweetviz library, the data is visualized in a variety of ways in an HTML file.
    """
    
    #EDA using Autoviz
    sweet_report = sv.analyze([DataFrame, "DataFrame"]) #Or just type the name of the dataframe

    #Saving results to HTML file
    sweet_report.show_html('sweet_report.html')

    return