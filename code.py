import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

y = df_train.SalePrice
X = df_train.drop(columns = ['SalePrice'])

# Select categorical columns with cardinality less than 10 and the missing value is not more than 1000
categorical_cols = [cname for cname in X.columns if (X[cname].nunique() < 10) & 
                    (X[cname].dtype == 'object') & 
                    (X[cname].isnull().sum() <= 1000)]

# Select numerical columns 
numerical_cols = [cname for cname in X.columns if (X[cname].dtype in ['int64','float64']) &
                 (X[cname].isnull().sum() <= 1000)]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
model_randomforest = RandomForestRegressor(n_estimators=200, random_state=0)

# XGBRegressor model
from xgboost import XGBRegressor
model_xgboost = XGBRegressor()

# Linear Regression
from sklearn.linear_model import LinearRegression
model_linreg = LinearRegression()

# AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
model_adab = AdaBoostRegressor(n_estimators=200, random_state=0)

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
model_grad = GradientBoostingRegressor()

from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model_randomforest)])

my_pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', model_xgboost)])

my_pipeline_linreg = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model_linreg)])

my_pipeline_adab = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model_adab)])

my_pipeline_grad = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model_grad)])
                                    
# define X and y for train and test 
X_train, X_valid, y_train, y_valid = train_test_split(X[my_cols], y, train_size=0.8, test_size=0.2, random_state=0)

my_pipeline_rf.fit(X_train, y_train)
my_pipeline_xgb.fit(X_train, y_train)
my_pipeline_linreg.fit(X_train, y_train)
my_pipeline_adab.fit(X_train, y_train)
my_pipeline_grad.fit(X_train, y_train)

preds_rf = my_pipeline_rf.predict(X_valid)
preds_xgb = my_pipeline_xgb.predict(X_valid)
preds_linreg = my_pipeline_linreg.predict(X_valid)
preds_adab = my_pipeline_adab.predict(X_valid)
preds_grad = my_pipeline_grad.predict(X_valid)

score_rf = mean_absolute_error(y_valid, preds_rf)
score_xgb = mean_absolute_error(y_valid, preds_xgb)
score_linreg = mean_absolute_error(y_valid, preds_linreg)
score_adab = mean_absolute_error(y_valid, preds_adab)
score_grad = mean_absolute_error(y_valid, preds_grad)

print('Score Random Forest Regressor: {}'.format(score_rf))
print('Score XGBoost Regressor: {}'.format(score_xgb))
print('Score Linear Regression: {}'.format(score_linreg))
print('Score AdaBoost Regressor: {}'.format(score_adab))
print('Score Gradient Boosting Regressor: {}'.format(score_grad))

# only choose best three
from sklearn.model_selection import cross_val_score

scores_rf = -1 * cross_val_score(my_pipeline_rf, X_train, y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')

scores_xgb = -1 * cross_val_score(my_pipeline_xgb, X_train, y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')

scores_grad = -1 * cross_val_score(my_pipeline_grad, X_train, y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')

print('MAE scores RF:\n', scores_rf)
print('MAE scores XGB:\n', scores_xgb)
print('MAE scores GB:\n', scores_grad)

print('Average MAE score RF (across experiments):')
print(scores_rf.mean())

print('Average MAE score XGB (across experiments):')
print(scores_xgb.mean())

print('Average MAE score GB (across experiments):')
print(scores_grad.mean())

# Choosing Gradient Boosting Regressor 
X_test = df_test[my_cols]

preds_gb = my_pipeline_grad.predict(X_test)

output = pd.DataFrame({'Id': df_test['Id'],
                       'SalePrice': preds_gb})
output.to_csv('submission_grad.csv', index=False)
