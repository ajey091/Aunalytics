import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

raw_data_train = pd.read_csv('au_train.csv')
raw_data_train.head()

X_train = raw_data_train.iloc[:,:14]
y_train = raw_data_train.iloc[:,-1]

le = LabelEncoder()
y_train = le.fit_transform(y_train)
# Exploratory data analysis

df = pd.DataFrame(y_train, columns=['Output_class'])
test = pd.concat([df,raw_data_train], axis=1)

raw_data_train = raw_data_train.drop(['fnlwgt', 'education'], axis=1)
X_train = raw_data_train.iloc[:,:12]
y_train = raw_data_train.iloc[:,-1]
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = pd.DataFrame(y_train,columns=['Output_class'])
print (raw_data_train.head())

cat = len(X_train.select_dtypes(include=['object']).columns)
num = len(X_train.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+',
      num, 'numerical', '=', cat+num, 'features')

# Loading and preprocessing testing data
raw_data_test = pd.read_csv('au_test.csv')
raw_data_test.head()
raw_data_test = raw_data_test.drop(['fnlwgt', 'education'], axis=1)

X_test = raw_data_test.iloc[:,:12]
y_test = raw_data_test.iloc[:,-1]
le = LabelEncoder()
y_test = le.fit_transform(y_test)
y_test = pd.DataFrame(y_test,columns=['Output_class'])

X_train['train'] = 1
X_test['train'] = 0

combined = pd.concat([X_train, X_test], axis=0)

combined = pd.get_dummies(combined, prefix_sep='_')
combined.head()

X_train = combined[combined['train'] == 1]
X_test = combined[combined['train'] == 0]
X_train.drop(['train'], axis=1, inplace=True)
X_test.drop(['train'], axis=1, inplace=True)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(max_depth=10, random_state=0, )
rfclf.fit(X_train_scaled, y_train)

feature_list = list(X_train_scaled.columns)
# Get numerical feature importances
importances = list(rfclf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

from matplotlib.pyplot import figure
figure(figsize=(12,14))
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values[:20], importances[:20], orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values[:20], feature_list[:20], rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.rcParams.update({'font.size': 28})
plt.tight_layout()
plt.savefig('feature1.png')

figure(figsize=(12,16))
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
figure(figsize=(10,10))
plt.plot(x_values[:20], cumulative_importances[:20], 'g-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=20, color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(x_values[:20], sorted_features[:20], rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')
plt.rcParams.update({'font.size': 22})
plt.tight_layout()
plt.savefig('feature2.png')

print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0] + 1)

rfacc = []
adaacc = []
for num in range(4,40):
    # Extract the names of the most important features
    important_feature_names = [feature[0] for feature in feature_importances[0:num]]
    # Find the columns of the most important features
    important_indices = [feature_list.index(feature) for feature in important_feature_names]
    # Create training and testing sets with only the important features
    important_train_features = X_train_scaled.iloc[:, important_indices]
    important_test_features = X_test_scaled.iloc[:, important_indices]
    # Sanity check on operations
    print('Important train features shape:', important_train_features.shape)
    print('Important test features shape:', important_test_features.shape)

    rfclf = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=1000)

    # Train the expanded model on only the important features
    rfclf.fit(important_train_features, y_train)
    # Make predictions on test data
    predictions = rfclf.predict(important_test_features)

    rfacc.append(accuracy_score(y_test, predictions))

    from sklearn.ensemble import AdaBoostClassifier
    adaclf = AdaBoostClassifier(n_estimators=1000)
    adaclf.fit( important_train_features, y_train)
    y_pred_ada = adaclf.predict(important_test_features)
    adaacc.append(accuracy_score(y_test, y_pred_ada))
    # print('AdaBoost accuracy=%.3f' % (accuracy_score(y_test, y_pred_ada)))


figure(figsize=(10,8))
plt.plot(range(4,40), rfacc, 'g', label="Random Forest")
plt.plot(range(4,40), adaacc, 'r', label = 'Adaboost')
plt.legend(loc="lower right")
plt.xlabel('Number of important features')
plt.ylabel('Accuracy')
plt.rcParams.update({'font.size': 28})
plt.tight_layout()
plt.savefig('feature3.png')
