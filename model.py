import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import joblib

# load data
df = pd.read_csv('chosen.csv', engine='python')

# insert mean to NA
imputer = SimpleImputer(strategy='mean')
numerical_df = df
filled_df = pd.DataFrame(imputer.fit_transform(numerical_df), columns=numerical_df.columns)
filled_df.to_csv('filled_chosen.csv', index=False)


X = filled_df.drop('aki', axis=1)
y = filled_df['aki']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# models
''''''
# GradientBoosting
cb_clf = CatBoostClassifier(iterations=100,
                           learning_rate=0.1,
                           depth=4,
                           loss_function='MultiClass',
                           verbose=False)
# grid search
cb_params = {'learning_rate': [0.03, 0.1, 0.3], 'depth': [4, 6, 10]}
grid_search_result = cb_clf.grid_search(cb_params, X=X_train, y=y_train, cv=5, plot=False)
best_cb_clf = cb_clf

# LogisticRegression
# grid search
params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['sag', 'saga']
}
log_reg = GridSearchCV(LogisticRegression(max_iter=10000), params, cv=5, verbose=1, n_jobs=-1)
log_reg.fit(X_train, y_train)
best_log_reg = log_reg.best_estimator_

''''''
# RandomForest
# grid search
params_2 = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]}
rf_clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), params_2, cv=5, verbose=1, n_jobs=-1)
rf_clf.fit(X_train, y_train)
best_rf_clf = rf_clf.best_estimator_

# save best_model
joblib.dump(best_cb_clf, 'best_cb_clf.joblib')
joblib.dump(best_log_reg, 'best_log_reg.joblib')
joblib.dump(best_rf_clf, 'best_rf_clf.joblib')






