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
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression



wind_ava = pd.read_csv('wind_ava.csv.gz', compression="gzip")

features = ["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "cape.13", "flsr.13", "fsr.13", "iews.13", "inss.13",
                          "lai_hv.13", "lai_lv.13", "u10n.13", "v10n.13", "stl1.13", "stl2.13", "stl3.13", "stl4.13", "sp.13", "p54.162.13", "p59.162.13", "p55.162.13"]



target= "energy"
fecha= wind_ava["datetime"]
#test van a ser las instancias de 2009 y 2008 para el test
test=0
for i in range (len(fecha)):
    if "2008" in fecha[i] or "2009" in fecha[i]:
        test+=1
df_seleccionado = wind_ava[features]
df_objetivo = wind_ava[target]
#Correlacion entre atributos
for i in features:
    print(i ,"\n")
    z=df_seleccionado.corrwith(df_seleccionado[i], method="pearson")
    print(z, "\n")
#Correlacion de los atributos con la columna objetivo (energy)
z=df_seleccionado.corrwith(df_objetivo, method="pearson")
print(z, "\n")
# Contar el número de valores faltantes por columna
cantidad_valores_faltantes_por_columna = df_seleccionado.isnull().sum()
#Calculo de columnas constantes
desviacion_estandar = df_seleccionado.std()
umbral_desviacion = 0.1
columnas_constantes = desviacion_estandar[desviacion_estandar < umbral_desviacion].index
print("Columnas Constantes:")
print(columnas_constantes)

#Prueba de modelos
minmax = MinMaxScaler()
standar = StandardScaler()
robust = RobustScaler()
clf = KNeighborsRegressor()
X = wind_ava[features]
y = wind_ava[target]
        
scalers = [
    minmax,
    standar,
    robust
]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test/(len(fecha)),shuffle=False)
for i in range (len(scalers)):
    pipe = Pipeline([
    ('scaler', scalers[i]), 
    ('knn', clf)]
)
    pipe.fit(X_train, y_train)
    y_test_pred=pipe.predict(X_test)
    r2 = pipe.score(X_test, y_test) #Lo que ha aprendido
    print(f'R2 del modelo: {r2}')
    mse = mean_squared_error(y_test, y_test_pred)**0.5
    print(mse)

#Selector de caracteristicas
preprocesador = RobustScaler()
estimador = KNeighborsRegressor()
pipeline = make_pipeline(preprocesador, estimador)
selector_caracteristicas = SequentialFeatureSelector(estimator=pipeline, n_features_to_select=2, scoring='accuracy', cv=5)
selector_caracteristicas.fit(X_train, y_train)
X_reducido = selector_caracteristicas.transform(X_train)
caracteristicas_seleccionadas = selector_caracteristicas.get_support()
nombres_caracteristicas_seleccionadas = selector_caracteristicas.k_feature_names_out_
indices_caracteristicas_seleccionadas = selector_caracteristicas.k_feature_idx_

# Imprimir resultados
print("Características seleccionadas:", caracteristicas_seleccionadas)
print("Nombres de características seleccionadas:", nombres_caracteristicas_seleccionadas)
print("Índices de características seleccionadas:", indices_caracteristicas_seleccionadas)