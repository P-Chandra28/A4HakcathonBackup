import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV # Used for splitting data and finding best parameters
from imblearn.combine import SMOTEENN # Helps with imbalanced data
from sklearn.linear_model import LogisticRegression # A simple model
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier # More complex models
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay # Used to evaluate our model
from sklearn.preprocessing import StandardScaler # Used to scale numerical data
from sklearn.compose import ColumnTransformer # Helps apply different steps to different columns
import matplotlib.pyplot as plt # Used for plotting graphs


def load_data(file_path):
    """Loads data from a CSV file into a pandas DataFrame."""
    # Read the CSV file into a DataFrame, which is like a table
    df = pd.read_csv(file_path)
    return df

def perform_feature_engineering(df):
    """Creates new features from existing ones."""
    # Count how many diagnoses each patient has
    df['num_diagnoses'] = df[['diag_1','diag_2','diag_3']].count(axis=1)

    # Sum up the number of lab procedures and other procedures
    df['total_med_procedures'] = df['n_lab_procedures'] + df['n_procedures']

    # Simplify the age groups into broader categories
    def simplify_age_group(age_range):
        if pd.isna(age_range): return 'Other' # Handle missing values
        if age_range in ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)']:
            return 'Young'
        elif age_range in ['[50-60)','[60-70)','[70-80)']:
            return 'Middle-aged'
        else:
            return 'Senior'
    df['age_group_simplified'] = df['age'].apply(simplify_age_group)


    return df

def preprocess_data(df, target_col):
    """PRepares the data for the model."""
    # Remove columns that are not needed for the model
    df = df.drop(['age', 'diag_1', 'diag_2', 'diag_3'], axis=1)
    # Remove the 'Unnamed: 0' column if it exists (sometimes created when saving data)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Simplify the medical specialty by keeping the most common ones and grouping the rest
    top_n_specialties = df['medical_specialty'].value_counts().nlargest(10).index.tolist()
    df['medical_specialty'] = df['medical_specialty'].apply(lambda x: x if x in top_n_specialties else 'Other')

    # Define which columns are numbers and which are categories
    categorical_cols = ['medical_specialty','A1Ctest','change','diabetes_med','age_group_simplified','glucose_test']
    numerical_cols = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'num_diagnoses', 'total_med_procedures','n_emergency', 'n_inpatient', 'n_outpatient']

    # Convert categorical columns into numerical ones using one-hot encoding (creating new columns for each category)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Convert the target column ('readmitted') into 0s and 1s
    df[target_col] = df[target_col].map({'no':0,'yes':1})

    # Clean up column names (replace special characters)
    df.columns = [str(c).replace('[','').replace(']','').replace('<','').replace(',','')
                  .replace('(','').replace(')','').replace(' ','_') for c in df.columns]

    # Separate the data into features (X) and the target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Print data types of X
    print("Data types of X before SMOTEENN:")
    print(X.dtypes)


    return X, y, numerical_cols

def train_and_evaluate_model(X, y, numerical_cols):
    """Trains and evaluates the model using cross-validation."""
    # Set up cross-validation to test the model on different parts of the data
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    f1_scores, roc_scores, acc_scores = [], [], []
    last_fold_results = {}

    # Create a preprocessor to scale numerical data and leave others as they are
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols)], remainder='passthrough')

    # Handle imbalanced data by oversampling the minority class and cleaning up
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    print("Resampled shape:", X_resampled.shape, y_resampled.shape)

    # Loop through each split of the data
    for fold,(train_idx,test_idx) in enumerate(rkf.split(X_resampled,y_resampled),1):
        print(f"\n--- Fold {fold} ---")
        # Split the data into training and testing sets for this fold
        X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
        y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]

        # Apply preprocessing (scaling) to the training and testing data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Set up and train the base models
        log_clf = LogisticRegression(solver='liblinear', max_iter=2000, random_state=42)
        log_clf.fit(X_train_processed,y_train)

        rf_clf = RandomForestClassifier(random_state=42)
        # Use GridSearchCV to find the best parameters for the Random Forest model
        grid_rf = GridSearchCV(rf_clf, {'n_estimators':[100],'max_depth':[10,20]}, cv=3, scoring='f1', n_jobs=-1)
        grid_rf.fit(X_train_processed,y_train)
        best_rf = grid_rf.best_estimator_ # Get the best Random Forest model

        # Train the AdaBoost model
        ada_clf = AdaBoostClassifier(random_state=42)
        # Use GridSearchCV to find the best parameters for the AdaBoost model
        grid_ada = GridSearchCV(ada_clf, {'n_estimators':[100],'learning_rate':[0.1,0.5]}, cv=3, scoring='f1', n_jobs=-1)
        grid_ada.fit(X_train_processed,y_train)
        best_ada = grid_ada.best_estimator_ # Get the best AdaBoost model

        # Set up and train the Stacking Classifier (combines the base models)
        stack_clf = StackingClassifier(estimators=[('lr',log_clf),('rf',best_rf),('ada',best_ada)],
                                       final_estimator=LogisticRegression(), cv=3, n_jobs=-1)
        stack_clf.fit(X_train_processed,y_train)
        feature_names = X_train.columns.tolist()

        # Find the best probability threshold for classification
        y_probs = stack_clf.predict_proba(X_test_processed)[:,1] # Get probabilities for the positive class
        thresholds = np.arange(0.1,0.9,0.01) # Test different thresholds
        best_thresh, best_f1 = 0.5, 0
        for t in thresholds:
            # Calculate F1 score for each threshold
            f1 = f1_score(y_test,(y_probs>=t).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, t # Keep track of the best threshold and F1 score

        # Make predictions using the best threshold
        y_pred = (y_probs>=best_thresh).astype(int)
        # Calculate evaluation metrics for this fold
        fold_acc = accuracy_score(y_test,y_pred)
        fold_roc_auc = roc_auc_score(y_test,y_probs)
        print(f"Fold {fold} -> Acc: {fold_acc:.4f}, F1: {best_f1:.4f}, ROC-AUC: {fold_roc_auc:.4f}")

        # Store the metrics for this fold
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

    # Print the average metrics across all folds
    print("\n Average F1:",np.mean(f1_scores))
    print(" Average ROC-AUC:",np.mean(roc_scores))
    print(" Average Accuracy:",np.mean(acc_scores))

    return np.mean(f1_scores), np.mean(roc_scores), np.mean(acc_scores), last_fold_results


def visualize_results(y_true, y_pred, y_probs):
    """Generates plots and calculates metrics to understand the model's performance."""

    # Print a detailed report of the model's performance
    print("\nClassification Report (Last Fold):\n", classification_report(y_true, y_pred, target_names=['No', 'Yes']))

    # Plot the Confusion Matrix to see how many predictions were correct and incorrect
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(cmap=plt.cm.Greens) # Use a green color map
    plt.title("Confusion Matrix (Last Fold)")
    plt.show()

    # Calculate and print the Mean Squared Error (MSE) (often used for regression, but included here)
    mse = mean_squared_error(y_true, y_pred)
    print(f"\n Mean Squared Error (Last Fold): {mse:.4f}")

    # Plot the ROC Curve to visualize the trade-off between true positive rate and false positive rate
    roc_auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_disp.plot(color='darkorange') # Use an orange color for the curve
    plt.title("ROC Curve (Last Fold)")
    plt.show()

    # Calculate and print the error rate (proportion of incorrect predictions)
    errors = y_true != y_pred
    error_rate = errors.mean()
    print(f"\nError Rate (Last Fold): {error_rate:.4f}")

    # More detailed error analysis could be done here, like plotting false positives vs. false negatives


# Main part of the script
file_path = 'notebook\data\hospital_knnimputer.csv' # Path to the data file
target_column = 'readmitted' # The column we want to predict

# 1. Load the data from the CSV file
df = load_data(file_path)

# 2. Create new features from the existing data
df = perform_feature_engineering(df)

# 3. Prepare the data for the machine learning model
X, y, numerical_cols = preprocess_data(df, target_column)

# 4. Train the model and see how well it performs
avg_f1, avg_roc_auc, avg_acc, last_fold_results = train_and_evaluate_model(X, y, numerical_cols)

# 5. Show the results using plots and metrics
visualize_results(last_fold_results['y_true'], last_fold_results['y_pred'], last_fold_results['y_probs'])