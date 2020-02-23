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

corrmat =  test.corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corrmat, vmax=.7, square=True)
plt.show()


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
y_pred_baseline = rfclf.predict(X_test_scaled)

print('Random Forest: Baseline accuracy=%.3f' % (accuracy_score(y_test, y_pred_baseline)))

rf_probs = rfclf.predict_proba(X_test_scaled)
rf_probs = rf_probs[:, 1]

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20,  50, 80],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [500, 1000]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid,
                          cv = 10, n_jobs = -1, verbose = 1,refit = True)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, np.ravel(y_train))
print (grid_search.best_params_)

rfclf = RandomForestClassifier(max_depth=20, random_state=1, n_estimators=1000)
rfclf.fit(X_train_scaled, y_train)
y_pred_gs = rfclf.predict(X_test_scaled)

print('Random Forest: Grid Search accuracy=%.3f' % (accuracy_score(y_test, y_pred_gs)))

from sklearn.ensemble import AdaBoostClassifier
adaclf = AdaBoostClassifier(n_estimators=1000)
adaclf.fit( X_train_scaled, y_train)
y_pred_ada = adaclf.predict(X_test_scaled)
print('AdaBoost accuracy=%.3f' % (accuracy_score(y_test, y_pred_ada)))


from sklearn.ensemble import GradientBoostingClassifier
gbclf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,max_depth=1, random_state=0).fit(X_train_scaled, y_train)
print ('Gradient Boosting classifier accuracy=%.3f' %(gbclf.score(X_test_scaled, y_test)))

