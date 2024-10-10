# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:13:21 2024

@author: User
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, balanced_accuracy_score
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree,svm
import shap
from collections import Counter
from sklearn.metrics import accuracy_score
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import time
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")




############  Credit card dataset
df = pd.read_excel('E:\\DataS Projects\\Marketing\\default of credit card clients.xls', engine='xlrd', header=None)
df.columns = df.iloc[1]
df = df.drop(index=[0, 1])
df.reset_index(drop=True, inplace=True)
df = df.drop(df.columns[0], axis=1)

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

#Convert categorical variables to dummy variables in both train and test datasets
X = pd.get_dummies(df.drop('default payment next month', axis=1), columns=categorical_features)
y= df['default payment next month']
y = y.astype(int)
columns_to_convert = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
X.fillna(0, inplace=True)  # or X1.dropna(inplace=True)
# Convert specified columns to numeric (int), coercing errors to NaN
for column in columns_to_convert:
    X[column] = pd.to_numeric(X[column], errors='coerce')
X = X.astype({col: 'int' for col in X.select_dtypes(include=['bool']).columns})

########### Bank marketing dataset

df = pd.read_csv ('E:\\DataS Projects\\Marketing\\PreprocessedBank.csv', header=None)
df.columns = df.iloc[0]  # The second row becomes the column names
df = df.drop(0)
df = df.iloc[:, 1:]
# Reset the index to ensure proper indexing
df = df.reset_index(drop=True)
X = df.iloc[:, 0:df.shape[1] - 1]  
y = df.iloc[:, df.shape[1] - 1]  
y = y.astype(int)
X1=X
X1.columns = X1.columns.astype(str)
for col in X1.columns:
    X1[col] = pd.to_numeric(X1[col], errors='coerce')

# Check for any remaining NaNs
print("Number of NaNs in features: ", X1.isna().sum().sum())

# If there are NaN values, you may want to handle them (e.g., fill, drop, etc.)
# Example: Fill NaNs with the mean of each column
X1.fillna(X.mean(), inplace=True)
###############################

#SRBCT-4c
df = pd.read_excel ('E:\DataS Projects\DataS\dataset2\SRBCT.xlsx', header=None)
df.iloc[:,df.shape[1]-1].replace({4:0},inplace=True)
X = df.iloc[:, 0:df.shape[1] - 1]  
y = df.iloc[:, df.shape[1] - 1]  



######## Binary & Multiclass datasets: SHAP-based filter
clf = xgboost.XGBClassifier()
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X1, y)

t0=time.time()
# Initialize the SHAP explainer with your model and binary dataset
explainer = shap.Explainer(clf, X1)
# compute SHAP values
shap_values = explainer(X1)
# Initialize the SHAP explainer with your model and multiclass dataset
# Aggregate the absolute SHAP values across all classes

if isinstance(shap_values, shap.Explanation):  # Check if it's a SHAP explanation object
    shap_values = shap_values.values  # Extract the actual SHAP values

if len(shap_values.shape) > 2:  # For multiclass SHAP values
    shap_values_mean = np.mean(np.abs(shap_values), axis=2)  # Mean across classes
else:
    shap_values_mean = np.abs(shap_values)  # For binary classification or regression
t1=time.time()
print(round(t1-t0))
# Now, we can create the bar plot
shap.summary_plot(shap_values_mean, X1, plot_type='bar', max_display=10, color='red', show=False)
# shap.summary_plot(shap_values_top,X1_top, plot_type='bar', max_display=10, color='red', show=False)
plt.title("SRBCT", fontsize=26)
plt.show()




#########Create the corresponding dataset from top 10 SHAP that handles Multiclass and binary#############
# Calculate mean absolute SHAP values based on the shape of shap_values
if len(shap_values.shape) > 2:  # Multiclass case
    mean_shap_values = np.abs(shap_values).mean(axis=2)  # Mean across classes
    # mean_shap_values = np.abs(shap_values).mean(axis=2).mean(axis=0)  # Mean across classes and samples
else:  # Binary case
    mean_shap_values = np.abs(shap_values)  # Mean across samples

# Check the shape of mean_shap_values
print("Shape of mean_shap_values after averaging:", mean_shap_values.shape)
# Now we need to reshape it to be 1D for feature importance
mean_shap_values= np.mean(mean_shap_values, axis=0)  # Take mean across samples to get 1D

# Create DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X1.columns,
    'Mean SHAP Value': mean_shap_values
})

# Sort the DataFrame by Mean SHAP Value
feature_importance_df = feature_importance_df.sort_values(by='Mean SHAP Value', ascending=False)

# Get the top 10 features
top_features = feature_importance_df.head(10)

# Print the top features
print("Top Features:")
print(top_features)
top_feature_names = top_features['Feature'].values  # Get the names of the top features
X_top = X1[top_feature_names]  # Filter X1 to only include the top features
############Accuracy of SHAP with top 10 best features##################

model = tree.DecisionTreeClassifier(random_state=42)
# model = GaussianNB()

# Perform 5-fold stratified cross-validation on the training data
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, X_top, y, cv=skf, scoring='accuracy')

y_pred = cross_val_predict(model, X_top, y, cv=skf)

# Calculate precision, recall, and F-score
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
fscore = f1_score(y, y_pred, average='macro')

# Output cross-validation scores and metrics
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
average_accuracy = cv_scores.mean()
rounded_average_accuracy = round(average_accuracy, 2)  # Round the average accuracy
print(f'Average Accuracy: {rounded_average_accuracy}')

# Output precision, recall, and F-score
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F-score: {fscore:.4f}')
##############################################################
#############################################################

################ Majority voting of top 10 best features########


# Get the top features
top_features = feature_importance_df.head(10)

# Create a list to store accuracies
y_hats = []
clf = tree.DecisionTreeClassifier(random_state=42)
# clf = GaussianNB()

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop through the number of top features to include
for i in range(1, len(top_features) + 1):
    selected_feature_names = top_features['Feature'].values[:i]
    
    # Select only the top features from X1
    X_subset = X1[selected_feature_names]
    
    # Initialize the Decision Tree model
    
    # Use cross_val_predict to get predictions for each instance in the dataset
    y_hat = cross_val_predict(clf, X_subset, y, cv=skf, method='predict')
    
    # Append the predictions to y_hats list
    y_hats.append(y_hat)




# Majority Voting
# Convert list of predictions to a NumPy array for easier manipulation
y_hats_array = np.array(y_hats).T  # Shape: (num_features, num_samples)

transposed_y_hats = np.array(y_hats).T

# Array to hold final predictions
final_predictions = []

for sample_predictions in transposed_y_hats:
    # Count the occurrences of each class
    count = Counter(sample_predictions)
    
    # Determine the class with the highest count
    most_common_class = count.most_common(1)[0][0]  # Get the class with the highest count
    
    final_predictions.append(most_common_class)

# Convert final predictions to a numpy array
final_y_hat = np.array(final_predictions)

# Calculate accuracy
accuracy = accuracy_score(y, final_y_hat)
rounded_accuracy = round(accuracy, 2)
print("Final Accuracy:", rounded_accuracy)


# Calculate precision, recall, and F-score
precision = precision_score(y, final_y_hat, average='macro')  # Adjust averaging method if necessary
recall = recall_score(y, final_y_hat, average='macro')        # Adjust averaging method if necessary
fscore = f1_score(y, final_y_hat, average='macro')            # Adjust averaging method if necessary

# Round the metrics to two decimal places
rounded_precision = round(precision, 2)
rounded_recall = round(recall, 2)
rounded_fscore = round(fscore, 2)

# Output rounded precision, recall, and F-score
print(f'Final Rounded Precision: {rounded_precision:.2f}')
print(f'Final Rounded Recall: {rounded_recall:.2f}')
print(f'Final Rounded F-score: {rounded_fscore:.2f}')


################# Mutual Information
from sklearn.feature_selection import mutual_info_classif

# Assuming X is your feature matrix and y is the target variable for classification
# Calculate mutual information for each feature with the target variable
t0=time.time()
mutual_info = mutual_info_classif(X, y, random_state=42)
t1=time.time()
print(round(t1-t0))

# Get indices of top features based on mutual information
limit = 10
top_indices = (-mutual_info).argsort()[:limit]  # Using negative values for descending order
top_feature_names = X.columns[top_indices]

Xn=X[top_feature_names]


model = tree.DecisionTreeClassifier(random_state=42)

# Perform 5-fold stratified cross-validation on the training data
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, Xn, y, cv=skf, scoring='accuracy')

y_pred = cross_val_predict(model, Xn, y, cv=skf)

# Calculate precision, recall, and F-score
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
fscore = f1_score(y, y_pred, average='macro')

# Output cross-validation scores and metrics
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
average_accuracy = cv_scores.mean()
rounded_average_accuracy = round(average_accuracy, 2)  # Round the average accuracy
print(f'Average Accuracy: {rounded_average_accuracy}')

# Output precision, recall, and F-score
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F-score: {fscore:.4f}')

#####################ReliefF
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import pandas as pd
X_array = X.values
X_array = X_array.astype(float)
limit = 10
t0=time.time()
relieff_selector = ReliefF(n_features_to_select=10, n_neighbors=100)
X_relief = relieff_selector.fit_transform(X_array, y)
t1=time.time()
print(round(t1-t0))


Xn=X[top_feature_names]


model = tree.DecisionTreeClassifier(random_state=42)

# Perform 5-fold stratified cross-validation on the training data
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, Xn, y, cv=skf, scoring='accuracy')

y_pred = cross_val_predict(model, Xn, y, cv=skf)

# Calculate precision, recall, and F-score
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
fscore = f1_score(y, y_pred, average='macro')

# Output cross-validation scores and metrics
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
average_accuracy = cv_scores.mean()
rounded_average_accuracy = round(average_accuracy, 2)  # Round the average accuracy
print(f'Average Accuracy: {rounded_average_accuracy}')

# Output precision, recall, and F-score
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F-score: {fscore:.4f}')
############ sample code for creating the bar charts in the paper

import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Precision', 'Recall', 'Fscore']
before = [0.83, 0.83, 0.83]
after = [0.94, 0.91, 0.92]

# X-axis positions
x = np.arange(len(labels))

# Adjusted bar width
width = 0.25  # Thinner bars

# Create the bar plot
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, before, width, label='Before applying SHAP-IEL', color='blue')
bars2 = ax.bar(x + width/2, after, width, label='After applying SHAP-IEL', color='orange')

# Add labels and title
ax.set_ylabel('Percentage')
ax.set_title('SRBCT', fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Set y-axis limit
ax.set_ylim(0.7, 1)  # Set y limit from 0 to 1

# Display percentage values on top of the bars
def add_percentage_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_percentage_labels(bars1)
add_percentage_labels(bars2)

# Show the plot
plt.tight_layout()
plt.show()
