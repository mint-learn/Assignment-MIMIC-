import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# load data
raw = pd.read_csv('sph6004_assignment1_data.csv', engine='python')

# Gradient Boosting select features
raw['gender'] = raw['gender'].map({'F': 0, 'M': 1})
raw = raw.drop('race', axis=1)
X = raw.drop('aki', axis=1)
y = raw['aki']
split_ratio = 0.8
raw = raw.sample(frac=1, random_state=42).reset_index(drop=True)
split_point = int(len(raw) * split_ratio)

# train test data
train_df = raw[:split_point]
test_df = raw[split_point:]

X_train = train_df.drop('aki', axis=1)
y_train = train_df['aki']
X_test = test_df.drop('aki', axis=1)
y_test = test_df['aki']

# train classifier
model = CatBoostClassifier(iterations=1000,
                           learning_rate=0.1,
                           depth=4,
                           loss_function='MultiClass',
                           verbose=False)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# calculate importance
feature_importances = model.get_feature_importance()
feature_names = X_train.columns
features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

print(features_df)

# choosing features
threshold = features_df['Importance'].mean()
important_features = features_df[features_df['Importance'] > threshold]
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

model.fit(X_train_important, y_train, eval_set=(X_test_important, y_test))

# write to doc
important_features = features_df[features_df['Importance'] > features_df['Importance'].mean()]['Feature'].tolist()
important_features.append('aki')
important_features_data = raw[important_features]
important_features_data.to_csv('chosen.csv', index=False)

