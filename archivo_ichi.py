import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


wind_ava = pd.read_csv('wind_ava.csv.gz', compression="gzip")

features = ["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "cape.13", "flsr.13", "fsr.13", "iews.13", "inss.13",
                          "lai_hv.13", "lai_lv.13", "u10n.13", "v10n.13", "stl1.13", "stl2.13", "stl3.13", "stl4.13", "sp.13", "p54.162.13", "p59.162.13", "p55.162.13"]
target= "energy"

df_seleccionado = wind_ava[features]

desviacion_estandar = df_seleccionado.std()

umbral_desviacion = 0.1

columnas_constantes = desviacion_estandar[desviacion_estandar < umbral_desviacion].index

print("Columnas Constantes:")
print(columnas_constantes)

target = "energy"

X = wind_ava[features]
y = wind_ava[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17,shuffle=False)


knn_regressor = KNeighborsRegressor(n_neighbors=3)

knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn_regressor.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn_regressor.score(X_test, y_test)))
print(f"Error cuadrático medio: {mse}")

features1 = ["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "cape.13", "iews.13", "inss.13",
                          "lai_hv.13", "lai_lv.13", "u10n.13", "v10n.13", "stl1.13", "stl2.13", "stl3.13", "stl4.13", "sp.13", "p54.162.13", "p59.162.13", "p55.162.13"]
target1= "energy"

df_seleccionado1 = wind_ava[features1]

desviacion_estandar1 = df_seleccionado1.std()

umbral_desviacion1 = 0.01

columnas_constantes1 = desviacion_estandar1[desviacion_estandar1 < umbral_desviacion1].index

print("Columnas Constantes:")
print(columnas_constantes1)

target = "energy"

X = wind_ava[features1]
y = wind_ava[target1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, shuffle=False)

knn_regressor = KNeighborsRegressor(n_neighbors=3)

knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn_regressor.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn_regressor.score(X_test, y_test)))
print(f"Error cuadrático medio: {mse}")
