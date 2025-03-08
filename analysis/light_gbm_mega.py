import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV

#reading the excel dataset
df = pd.read_excel("C:\\Users\\ritwi\\OneDrive\\Desktop\\mpce-ml\\data\\train_df.xlsx")
df_test = pd.read_excel("C:\\Users\\ritwi\\OneDrive\\Desktop\\mpce-ml\\data\\test_mpce.xlsx")



df_main = pd.concat([df, df_test], ignore_index=True)
print(df_main.head())




scaler = StandardScaler()
X = df_main.drop(columns=["TotalExpense"])
numerical_cols =['Total_year_of_education_completed_HH_head', 'HH_Size (For FDQ)']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
y = df_main['TotalExpense']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)



lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)
params = {
    'objective': 'regression',  
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

gbm_mega = lgb.train(
    params, lgb_train, num_boost_round = 1000, valid_sets = [lgb_train,lgb_test], callbacks = [lgb.early_stopping(stopping_rounds=100)]
)
y_test_pred = gbm_mega.predict(X_test, num_iteration=gbm_mega.best_iteration)
r2 = r2_score(y_test, y_test_pred )
print(r2)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE:", rmse)

