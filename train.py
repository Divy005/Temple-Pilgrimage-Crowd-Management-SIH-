

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')


X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')['darshans']
y_test = pd.read_csv('y_test.csv')['darshans']
dates_test = pd.read_csv('dates_test.csv')['date']


X = pd.concat([X_train, X_test])
y = pd.concat([pd.Series(y_train, name='darshans'), pd.Series(y_test, name='darshans')])


param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15]
}
rf_model = RandomForestRegressor(random_state=42)
tscv = TimeSeriesSplit(n_splits=5)
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
print("Random Forest Best Parameters:", grid_search_rf.best_params_)

best_rf_model.fit(X_train, y_train)
predictions = best_rf_model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")


plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual', color='blue')
plt.plot(dates_test, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Darshans')
plt.title('Actual vs Predicted Darshans (Random Forest)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_rf_model.feature_importances_})
importances = importances.sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(importances)


mses, maes = [], []
for train_idx, test_idx in tscv.split(X):
    model = RandomForestRegressor(**grid_search_rf.best_params_, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    mses.append(mean_squared_error(y.iloc[test_idx], preds))
    maes.append(mean_absolute_error(y.iloc[test_idx], preds))
print("\nCross-Validation Results:")
print(f"Average CV MSE: {sum(mses)/len(mses)}")
print(f"Average CV MAE: {sum(maes)/len(maes)}")


joblib.dump(best_rf_model, 'random_forest_darshans.pkl')
print("Model saved as 'random_forest_darshans.pkl'")