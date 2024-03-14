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

wind_ava = pd.read_csv('wind_ava.csv.gz', compression="gzip")

features = ["t2m.13", "u10.13", "v10.13", "u100.13", "v100.13", "cape.13", "flsr.13", "fsr.13", "iews.13", "inss.13",
                          "lai_hv.13", "lai_lv.13", "u10n.13", "v10n.13", "stl1.13", "stl2.13", "stl3.13", "stl4.13", "sp.13", "p54.162.13", "p59.162.13", "p55.162.13"]
target= "energy"
fecha= wind_ava["datetime"]
#X van a ser las instancias de 2009
x=0
for i in range (len(fecha)):
    if "2008" in fecha[i] or "2009" in fecha[i]:
        x+=1
df_seleccionado = wind_ava[features]
# Contar el número de valores faltantes por columna
cantidad_valores_faltantes_por_columna = df_seleccionado.isnull().sum()
#Calculo de columnas constantes
desviacion_estandar = df_seleccionado.std()
umbral_desviacion = 0.1
columnas_constantes = desviacion_estandar[desviacion_estandar < umbral_desviacion].index
print("Columnas Constantes:")
print(columnas_constantes)
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

target = "energy"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=x/(len(fecha)),shuffle=False)
for i in range (len(scalers)):
    pipe = Pipeline([
    ('scaler', scalers[i]), 
    ('knn', clf)]
)
    pipe.fit(X_train, y_train)
     #mape
    r2 = pipe.score(X_test, y_test) #Lo que ha aprendido
    print(f'Accuracy del modelo: {r2}')

    



# Entrenar el modelo con el Pipeline


#mse = mean_squared_error(y_test, y_pred)**0.5
#print (mse)