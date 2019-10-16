import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Importing the dataset
dataset = pd.read_csv('Clients.csv')
y = dataset['Exited']
X = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)


# Checking categorical variables
categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']
data_describe = X.describe(include=[object])
binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(f'Categorical_columns: {categorical_columns}')
print(f'Binary_columns: {binary_columns}')
print(f'Non-binary_columns: {nonbinary_columns}')

# Encoding binary columns
for binary_column in binary_columns:
  labelencoder_binary = LabelEncoder()
  X[binary_column] = labelencoder_binary.fit_transform(X[binary_column])    

# Encoding non-binary columns
for nonbinary_column in nonbinary_columns:
  labelencoder_nonbinary = LabelEncoder()
  X[nonbinary_column] = labelencoder_nonbinary.fit_transform(X[nonbinary_column])    
  
transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoding", 
         OneHotEncoder(
                categories='auto' , 
                drop='first' # collinear features may cause problems, when feeding the resulting data into a neural network or an unregularized regression
         ), 
        [X.columns.get_loc(c) for c in nonbinary_columns])
    ], remainder='passthrough'
)
X = transformer.fit_transform(X)  


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Building and tuning ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier) # use a scikit learn wrapper to include K-fold cross validation

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']} #Adaptive Moment Estimation, RMSProp - compute adaptive learning rates for each parameter
              
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
