import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb

def train_model():
    # 1. Load the splits
    X_train = pd.read_csv('Xtrain.csv')
    X_test = pd.read_csv('Xtest.csv')
    y_train = pd.read_csv('ytrain.csv').values.ravel()
    y_test = pd.read_csv('ytest.csv').values.ravel()
    
    # 2. Identify numerical and categorical features
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # 3. Create preprocessing pipeline
    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown='ignore'), cat_cols)
    )
    
    # 4. Define model pipeline
    model_pipeline = make_pipeline(
        preprocessor,
        xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    )
    
    # 5. Define hyperparameter grid for tuning
    param_grid = {
        'xgbclassifier__n_estimators': [50, 100],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__learning_rate': [0.05, 0.1]
    }
    
    # 6. Set up MLflow Tracking
    mlflow.set_experiment("Tourism_Wellness_Prediction")
    
    with mlflow.start_run():
        print("Tuning hyperparameters via Grid Search...")
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # 7. Evaluate on training and testing data
        y_pred = best_model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        # 8. Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        print("\n--- Best Parameters found ---")
        print(grid_search.best_params_)
        
        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # 9. Ensure deployment directory exists and save best model
        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, "tourism_project/deployment/best_model.joblib")
        print("\nBest model successfully saved to tourism_project/deployment/best_model.joblib")

if __name__ == '__main__':
    train_model()
