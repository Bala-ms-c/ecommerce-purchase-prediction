import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('data/raw/online_shoppers_intention.csv')

# 1. Load the preprocessor
preprocessor = joblib.load('C:/Users/BALA/Downloads/Palindrome/ecommerce_purchase_prediction/models/preprocessor.pkl')

# 2. Preprocess the data using the loaded preprocessor (train and test data)
# The preprocessor applies outlier capping, missing value imputation, encoding, etc.
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Apply the preprocessing transformations (both for training and test)
X_processed = preprocessor.transform(X)  # This applies the transformations for prediction

# Split the processed data into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Calculate scale_pos_weight as the ratio of the negative class to the positive class
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Define the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # binary classification
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',  # can also try 'auc' for ROC-AUC metric
    use_label_encoder=False, # suppress warnings
    max_depth=6,  # increased depth
    learning_rate=0.1,  # adjusted learning rate
    n_estimators=200,  # you can try different numbers of trees as well
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Saving model now...")
# Save the trained model
joblib.dump(xgb_model, 'C:/Users/BALA/Downloads/Palindrome/ecommerce_purchase_prediction/models/xgb_model_new.pkl')
print("Model saved ✅")