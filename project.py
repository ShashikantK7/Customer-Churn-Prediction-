# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
# Assuming the dataset has columns like 'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'
data = pd.read_csv("customer_churn.csv")  # Replace with your dataset path
print(data.head())

# Data Preprocessing
# Drop unnecessary columns
data = data.drop(['customerID'], axis=1)

# Handling missing values (assuming 'TotalCharges' may have missing values)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.fillna(data.mean())  # Fill missing values with mean

# Encoding categorical variables
label_encoder = LabelEncoder()
data['Churn'] = label_encoder.fit_transform(data['Churn'])  # Encode 'Churn' as 0 and 1
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Splitting features (X) and target (y)
X = data.drop('Churn', axis=1)  # Features (everything except 'Churn')
y = data['Churn']  # Target (Churn column)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation of Logistic Regression
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Evaluation of Random Forest
print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
