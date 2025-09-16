import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.utilis import save_object
import os
import sys
import joblib

@dataclass
class ModelConfig:
    scaler_obj_file_path=os.path.join("artifacts","scaler.pkl")
    model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelConfig()

    
    def train_and_evaluate_model(self,X, y):
        try:
            logging.info("Model training has begun")
            rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            f1_scores, roc_scores, acc_scores = [], [], []
            last_fold_results = {}

            numerical_cols = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'num_diagnoses', 'total_med_procedures','n_emergency', 'n_inpatient', 'n_outpatient']
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols)], remainder='passthrough')
            logging.info("Preprocessor created for scaling numerical features")

            logging.info("Starting resampling using SMOTEENN")
            smote_enn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smote_enn.fit_resample(X=X, y=y)
            print("Resampled shape:", X_resampled.shape, y_resampled.shape)

            logging.info("Beginning cross-validation and model training")
            for fold,(train_idx,test_idx) in enumerate(rkf.split(X_resampled,y_resampled),1):
                print(f"\n--- Fold {fold} ---")
                
                X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
                y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]

               
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)


                log_clf = LogisticRegression(solver='liblinear', max_iter=2000, random_state=42)
                log_clf.fit(X_train_processed,y_train)

                rf_clf = RandomForestClassifier(random_state=42)
                grid_rf = GridSearchCV(rf_clf, {'n_estimators':[100],'max_depth':[10,20]}, cv=3, scoring='f1', n_jobs=-1)
                grid_rf.fit(X_train_processed,y_train)
                best_rf = grid_rf.best_estimator_ 

                ada_clf = AdaBoostClassifier(random_state=42)

                grid_ada = GridSearchCV(ada_clf, {'n_estimators':[100],'learning_rate':[0.1,0.5]}, cv=3, scoring='f1', n_jobs=-1)
                grid_ada.fit(X_train_processed,y_train)
                best_ada = grid_ada.best_estimator_ 


                stack_clf = StackingClassifier(estimators=[('lr',log_clf),('rf',best_rf),('ada',best_ada)],
                                            final_estimator=LogisticRegression(), cv=3, n_jobs=-1)
                stack_clf.fit(X_train_processed,y_train)
                feature_names = X_train.columns.tolist()
                logging.info("Feature names extracted")
                
                joblib.dump(feature_names, "artifacts/feature_names.pkl")

                save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=stack_clf
                )
                save_object(
                file_path=self.model_trainer_config.scaler_obj_file_path,
                obj=preprocessor
                )
                logging.info("Model pickle file saved to artifacts")

                y_probs = stack_clf.predict_proba(X_test_processed)[:,1]
                thresholds = np.arange(0.1,0.9,0.01) 
                best_thresh, best_f1 = 0.5, 0
                logging.info("Finding the best threshold based on F1 score")    

                for t in thresholds:
                    
                    f1 = f1_score(y_test,(y_probs>=t).astype(int))
                    if f1 > best_f1:
                        best_f1, best_thresh = f1, t 

                logging.info(f"Best threshold: {best_thresh}, Best F1: {best_f1}")
                
                y_pred = (y_probs>=best_thresh).astype(int)
                logging.info("Predictions made on the test set")    
                fold_acc = accuracy_score(y_test,y_pred)
                fold_roc_auc = roc_auc_score(y_test,y_probs)
                print(f"Fold {fold} -> Acc: {fold_acc:.4f}, F1: {best_f1:.4f}, ROC-AUC: {fold_roc_auc:.4f}")

                logging.info(f"Classification Report for fold {fold}:\n{classification_report(y_test,y_pred)}")
                f1_scores.append(best_f1)
                roc_scores.append(fold_roc_auc)
                acc_scores.append(fold_acc)

                # Store the results of the last fold for visualization
                if fold == rkf.get_n_splits():
                    last_fold_results = {
                        'y_true': y_test,
                        'y_pred': y_pred,
                        'y_probs': y_probs
                    }
            

            logging.info("Cross-validation and model training completed")
            print("\n--- Cross-Validation Results ---")
            print("\n Average F1:",np.mean(f1_scores))
            print(" Average ROC-AUC:",np.mean(roc_scores))
            print(" Average Accuracy:",np.mean(acc_scores))
            
            return np.mean(f1_scores), np.mean(roc_scores), np.mean(acc_scores), last_fold_results


        
        except Exception as e:
            raise CustomException(e,sys)