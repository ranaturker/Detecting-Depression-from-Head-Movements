import os  # Provides functions to interact with the operating system
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Library for numerical computations

# Importing machine learning tools
from sklearn.model_selection import LeaveOneOut, GridSearchCV, StratifiedKFold  # Cross-validation methods and grid search for hyperparameter tuning
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.metrics import classification_report, accuracy_score  # Evaluation metrics
from sklearn.preprocessing import StandardScaler  # Standardizes features by removing mean and scaling to unit variance
from imblearn.over_sampling import SMOTE  # Handles class imbalance through oversampling
from imblearn.pipeline import Pipeline  # Allows creating machine learning pipelines including preprocessing and model training
import warnings  # Handles warnings in the code

# -------------------------------------------------------------------------
# 1) Paths Configuration
# -------------------------------------------------------------------------
BASE_PATH = r"C:\Users\yagmu\PycharmProjects\capstone"  # Base directory for project files
Y_MATRIX_PATH = r"C:\Users\yagmu\PycharmProjects\capstone\y_matrix_modified.csv"  # Path to the target variable data
OUTPUT_PATH = os.path.join(BASE_PATH, 'svm_all_outputs_LOOCV.txt')  # Output file to save results

# -------------------------------------------------------------------------
# 2) Load y_matrix
# -------------------------------------------------------------------------
try:
    y_matrix = pd.read_csv(Y_MATRIX_PATH)  # Load target variable data
except FileNotFoundError:
    raise FileNotFoundError(f"y_matrix file not found at {Y_MATRIX_PATH}")  # Raise error if file not found

# Check if 'PHQ8_Binary' column exists in the dataset
if 'PHQ8_Binary' not in y_matrix.columns:
    raise ValueError("'PHQ8_Binary' column not found in y_matrix.")  # Raise error if column is missing

# -------------------------------------------------------------------------
# 3) Write header to the output file
# -------------------------------------------------------------------------
with open(OUTPUT_PATH, 'w') as f:
    f.write("SVM Training Results (Nested LOOCV + GridSearch) for All Matrices\n\n")  # Add header for output

# -------------------------------------------------------------------------
# 4) Create Leave-One-Out Cross-Validation (LOOCV) object
# -------------------------------------------------------------------------
loo = LeaveOneOut()  # Define LOOCV method for evaluation

# -------------------------------------------------------------------------
# 5) Helper Functions for Logging (Optional)
# -------------------------------------------------------------------------
def log_data_summary(f, index_, X, y):
    """Writes summary of X and y to the output file."""
    f.write(f"Index{index_} Data Summary:\n")
    f.write(f"X shape: {X.shape}\n")  # Log shape of features
    f.write("y distribution:\n")
    f.write(f"{y.value_counts()}\n")  # Log distribution of target variable
    f.write("X summary:\n")
    f.write(f"{X.describe()}\n\n")  # Log descriptive statistics of features

def log_unique_predictions(f, index_, y_pred):
    """Logs unique predictions for each index."""
    unique_preds = np.unique(y_pred)  # Find unique predictions
    f.write(f"Index{index_} Unique Predictions: {unique_preds}\n\n")  # Log predictions

# -------------------------------------------------------------------------
# 6) Hyperparameter Grid for Tuning
# -------------------------------------------------------------------------
param_grid = {
    'svm__C': [0.1, 1, 10, 100],  # Values for regularization parameter C
    'svm__gamma': ['scale', 0.01, 0.1, 1]  # Values for kernel coefficient gamma
}

# -------------------------------------------------------------------------
# 7) Loop through all index files
# -------------------------------------------------------------------------
for i in range(1, 289):  # Iterate through index files 1 to 288
    try:
        x_matrix_filename = f'Index{i}_combined_matrix.csv'  # File name for features
        x_matrix_path = os.path.join(BASE_PATH, x_matrix_filename)  # Path to feature file

        # Check if the file exists
        if not os.path.exists(x_matrix_path):
            raise FileNotFoundError(f"{x_matrix_path} does not exist.")

        # Load X matrix (features)
        x_matrix = pd.read_csv(x_matrix_path)

        # Check for mismatched sample sizes
        if y_matrix.shape[0] != x_matrix.shape[0]:
            raise ValueError(
                f"Sample size mismatch between y_matrix ({y_matrix.shape[0]}) "
                f"and {x_matrix_filename} ({x_matrix.shape[0]})."
            )

        # Combine target variable and features
        combined_data = pd.concat(
            [y_matrix.reset_index(drop=True), x_matrix.reset_index(drop=True)],
            axis=1
        )

        # Verify 'PHQ8_Binary' column exists
        if 'PHQ8_Binary' not in combined_data.columns:
            raise ValueError(
                f"'PHQ8_Binary' column not found after merging with {x_matrix_filename}."
            )

        # Split features and target variable
        X = combined_data.drop(columns=['PHQ8_Binary'])
        y = combined_data['PHQ8_Binary']

        # Check for missing values
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError(f"Missing values detected in {x_matrix_filename}.")

        # Log data summary (optional)
        with open(OUTPUT_PATH, 'a') as f:
            log_data_summary(f, i, X, y)

        print(f"Index{i}: Data Loaded Successfully. X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")

        # Initialize lists to store predictions and true values
        y_pred_all = []
        y_true_all = []

        # ---------------------------------------------------------------------
        # 8) Manual LOOCV: Train on n-1 samples, test on 1 sample
        # ---------------------------------------------------------------------
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Define pipeline with SMOTE, scaling, and SVC
            inner_pipeline = Pipeline([
                ('smote', SMOTE(random_state=i)),
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', class_weight='balanced'))
            ])

            # Perform grid search for hyperparameters
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=i)
            grid_search = GridSearchCV(estimator=inner_pipeline, param_grid=param_grid, scoring='accuracy', cv=inner_cv, n_jobs=-1, verbose=0)

            # Fit model and predict test sample
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.best_estimator_.predict(X_test)

            y_pred_all.extend(y_pred)
            y_true_all.extend(y_test)

        # Compute accuracy and classification report
        accuracy = accuracy_score(y_true_all, y_pred_all)
        report = classification_report(y_true_all, y_pred_all, zero_division=0)

        # Log results
        with open(OUTPUT_PATH, 'a') as f:
            f.write(f"Results for {x_matrix_filename}:\n")
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"{report}\n\n")

    except Exception as e:
        with open(OUTPUT_PATH, 'a') as f:
            f.write(f"Error processing {x_matrix_filename}: {str(e)}\n\n")

print(f"All LOOCV results saved to {OUTPUT_PATH}")
