# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
# Load training data which includes the churn labels
train_data = pd.read_csv("customer_churn_train.csv")  # Replace with your dataset path
print("Training Data Sample:")
print(train_data.head())

# Load testing data which does not include churn labels
test_data = pd.read_csv("customer_churn_test.csv")  # Replace with your dataset path
print("Testing Data Sample:")
print(test_data.head())

# Data Preprocessing
# Drop unnecessary columns from training data
train_data = train_data.drop(['customerID'], axis=1)

# Handle missing values in training data
train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce')
train_data = train_data.fillna(train_data.mean())  # Fill missing values with mean

# Encode categorical variables in training data
label_encoder = LabelEncoder()
train_data['Churn'] = label_encoder.fit_transform(train_data['Churn'])  # Encode 'Churn' as 0 and 1
for column in train_data.select_dtypes(include=['object']).columns:
    train_data[column] = label_encoder.fit_transform(train_data[column])

# Prepare features and target variable for training
X_train = train_data.drop('Churn', axis=1)  # Features
y_train = train_data['Churn']  # Target

# Train-test split (for training data validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_val)

# Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

# Evaluation of Logistic Regression on training-validation split
print("Logistic Regression Validation Results:")
print("Accuracy:", accuracy_score(y_val, y_pred_log_reg))
print(classification_report(y_val, y_pred_log_reg))

# Evaluation of Random Forest on training-validation split
print("Random Forest Validation Results:")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))

# Prepare and preprocess test data
test_data = test_data.drop(['customerID'], axis=1)
test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce')
test_data = test_data.fillna(test_data.mean())
for column in test_data.select_dtypes(include=['object']).columns:
    test_data[column] = label_encoder.transform(test_data[column])

# Feature scaling on test data
X_test = scaler.transform(test_data)

# Predict on the test data
test_pred_log_reg = log_reg.predict(X_test)
test_pred_rf = rf.predict(X_test)

# Save predictions to CSV (optional)
submission = pd.DataFrame({
    'customerID': test_data['customerID'],
    'Logistic_Regression_Prediction': test_pred_log_reg,
    'Random_Forest_Prediction': test_pred_rf
})
submission.to_csv('churn_predictions.csv', index=False)
print("Predictions saved to 'churn_predictions.csv'")
