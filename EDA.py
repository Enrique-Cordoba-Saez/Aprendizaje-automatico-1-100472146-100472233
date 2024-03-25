import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits



wind_ava = pd.read_csv('wind_ava.csv.gz', compression="gzip")

features = ["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "cape.13", "flsr.13", "fsr.13", "iews.13", "inss.13",
                          "lai_hv.13", "lai_lv.13", "u10n.13", "v10n.13", "stl1.13", "stl2.13", "stl3.13", "stl4.13", "sp.13", "p54.162.13", "p59.162.13", "p55.162.13"]

features_totales=[]
features_totales.append(features)
target= "energy"
fecha= wind_ava["datetime"]
#test van a ser las instancias de 2009 y 2008 para el test
X = wind_ava[features]
y = wind_ava[target]
# Contar el número de valores faltantes por columna
cantidad_valores_faltantes_por_columna = X.isnull().sum()
#Correlacion entre atributos
for i in features:
    corr = X.corrwith(X[i], method="pearson")
    high_correlations = corr[corr.abs() > 0.99]
    #if not high_correlations.empty:
        #print(f"Correlaciones altas para {i}:")
        #print(high_correlations)
#Correlacion de los atributos con la columna objetivo (energy)
correlacion_objetivo=X.corrwith(y, method="pearson")
print(correlacion_objetivo, "\n")
features_totales.append(["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "flsr.13", "fsr.13", "iews.13", "inss.13",
                        "lai_hv.13", "lai_lv.13", "u10n.13", "v10n.13", "stl1.13", "stl2.13", "stl3.13","sp.13", "p54.162.13", "p59.162.13"])
#Calculo de columnas constantes
desviacion_estandar = X.std()
umbral_desviacion = 0.1
columnas_constantes = desviacion_estandar[desviacion_estandar < umbral_desviacion].index
print("Columnas Constantes:", columnas_constantes )
features_sin_constantes=features
for i in columnas_constantes:
  features_sin_constantes.remove(i)
features_totales.append(features_sin_constantes)
#Division en test
test=0
for i in range (len(fecha)):
    if "2008" in fecha[i] or "2009" in fecha[i]:
        test+=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test/(len(fecha)),shuffle=False)

#Seleccion de columnas
k_best =SelectKBest(score_func=f_regression,k=12)  
fit= k_best.fit(X_train,y_train)   
selected_features = X_train.columns[k_best.get_support()]
features_totales.append(selected_features)
#De la tesis
features_totales.append(["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13",
                         "u10n.13", "v10n.13", "sp.13",])
print(selected_features)
#Prueba de modelos
minmax = MinMaxScaler()
standar = StandardScaler()
robust = RobustScaler()
clf = KNeighborsRegressor()
scalers = [
    minmax,
    standar,
    robust
]
nombres=["MinMax","Standar", "Robust"]
for i in features_totales:
    X = wind_ava[i]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test/(len(fecha)),shuffle=False)
    for i in range (len(scalers)):
        
        pipe = Pipeline([
        ('scaler', scalers[i]), 
        ('knn', clf)]
    )
        pipe.fit(X_train, y_train)
        y_test_pred=pipe.predict(X_test)
        r2 = pipe.score(X_test, y_test) #Lo que ha aprendido
        print(f'R2 del modelo: {r2} con ', nombres[i])
        mse = mean_squared_error(y_test, y_test_pred)**0.5
        print("Root mean squared error: ", mse)


