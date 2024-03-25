
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from lineartree import LinearTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import time
wind_ava = pd.read_csv('wind_ava.csv.gz', compression="gzip")

features = ["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "cape.13", "flsr.13", "fsr.13", "iews.13", "inss.13",
                          "lai_hv.13", "lai_lv.13", "u10n.13", "v10n.13", "stl1.13", "stl2.13", "stl3.13", "stl4.13", "sp.13", "p54.162.13", "p59.162.13", "p55.162.13"]
target= "energy"
X = wind_ava[features]
y = wind_ava[target]
fecha= wind_ava["datetime"]
test=0
for i in range (len(fecha)):
    if "2008" in fecha[i] or "2009" in fecha[i]:
        test+=1
k_best =SelectKBest(score_func=f_regression,k=12)  
fit= k_best.fit(X,y)   
selected_features = X.columns[k_best.get_support()]


#Creacion de regresores
knn_regressor=KNeighborsRegressor()
tree_regressor = LinearTreeRegressor(base_estimator=LinearRegression())
linear_regressor = LinearRegression()
lasso_regressor = Lasso() 
svm_regressor = SVR()
#mlp_regressor = MLPRegressor()  
minmax = MinMaxScaler()
standar = StandardScaler()
robust = RobustScaler()
scalers=[minmax,standar,robust]
modelos=[knn_regressor, linear_regressor,lasso_regressor,svm_regressor,tree_regressor]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test/len(fecha), shuffle=False)
for modelo in range (len(modelos)):
    for scaler in scalers:
        pipe = Pipeline([
            ('scaler',scaler), 
            ('knn', modelos[modelo])]) 
        start_time = time.time()
        pipe.fit(X_train, y_train)
        y_test_pred = pipe.predict(X_test)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecucion: {execution_time} segundos")
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
        print(f"RMSE of the model {modelos[modelo]}: {rmse}")
        r2 = r2_score(y_test, y_test_pred)
        print(r2)

scalers=[minmax,standar,robust]
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  
    'leaf_size': [20, 30, 40], 
    'p': [1, 2] 
}

param_grid_tree = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [ 6, 10],
    'min_samples_leaf': [3, 4]
}
param_grid_linear = {}
param_grid_lasso = {
    'alpha': [0.1, 1.0, 10.0],
    'max_iter': [10000,100000,1000000],
    'selection': ['cyclic', 'random']
}
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.01, 0.001]
}
"""param_grid_neuronas = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}
"""
#Transformamos los datos aqui
X_train = standar.fit_transform(X_train)
X_test = standar.transform(X_test)
param_grid=[param_grid_knn,param_grid_linear,param_grid_lasso, param_grid_svr,param_grid_tree]

for modelo in range (len(modelos)):
    start_time = time.time()
    grid_search = GridSearchCV( modelos[modelo], param_grid[modelo], scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Mejores parámetros",grid_search.best_params_, "del modelo ",modelos[modelo])
    print("  Mejor error (RMSE):", -grid_search.best_score_)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecucion: {execution_time} segundos") 