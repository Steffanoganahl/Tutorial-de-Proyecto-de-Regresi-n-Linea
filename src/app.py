from utils import db_connect
engine = db_connect()

# your code here

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

#Imported Necessary Libraries // Librerias necesarias para el proyecto. 

# Load Database / Cargar base de datos.
df = pd.read_csv('/workspaces/Tutorial-de-Proyecto-de-Regresi-n-Linea/data/raw/medical_insurance_cost.csv')
total_data = df 
total_data.head()

# Database features / Características de la base de datos
print(total_data.shape)
print(total_data.describe())
print(total_data.info())

#In this case we took "charges" as the study main variable / Se toma en este caso "charges" como la variable principal de estudio.
#We got 4 numerical variables (age,bmi,children,charges) and 3 categorical (sex, smoker, region) / Tenemos 4 variables numéricas (age,bmi,children,charges)
# y 3 categoricas (sex, smoker, region).

#Eliminate duplicates / Eliminar duplicados. 
total_data[total_data.duplicated(keep=False)]
total_data = total_data.drop_duplicates()
total_data.shape

#Check for null values / Verificar valores nulos.
total_data.isnull().sum()

#Determinate irrelevant features / Determinar características irrelevantes.
#In this case we don't have any irrelevant feature / En este caso no tenemos ninguna característica irrelevante.

fig, axis = plt.subplots(1, 3, figsize=(14, 5))

# Create Histogram / Crear histogramas
sns.histplot(ax = axis[0], data = total_data, x = "sex")
sns.histplot(ax = axis[1], data = total_data, x = "smoker")
sns.histplot(ax = axis[2], data = total_data, x = "region")

# Adjust layout / Ajustar el diseño
plt.tight_layout()

# Show the plot / Mostrar el gráfico
plt.show()

fig, axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})

sns.histplot(ax = axis[0, 0], data = total_data, x = "charges")
sns.boxplot(ax = axis[1, 0], data = total_data, x = "charges")

sns.histplot(ax = axis[0, 1], data = total_data, x = "age")
sns.boxplot(ax = axis[1, 1], data = total_data, x = "age")

sns.histplot(ax = axis[2, 0], data = total_data, x = "bmi")
sns.boxplot(ax = axis[3, 0], data = total_data, x = "bmi")

sns.histplot(ax = axis[2, 1], data = total_data, x = "children")
sns.boxplot(ax = axis[3, 1], data = total_data, x = "children")

# Adjust layout / Ajustar el diseño
plt.tight_layout()

# Show the plot / Mostrar el gráfico
plt.show()


# Numerical - Numerical Analysis / Análisis numérico - numérico

# Create subplot canvas / Crear lienzo de subgráficos
fig, axis = plt.subplots(4, 2, figsize = (10, 16))

# Create Plates / Crear gráficos
sns.regplot(ax = axis[0, 0], data = total_data, x = "age", y = "charges")
sns.heatmap(total_data[["charges", "age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = total_data, x = "bmi", y = "charges").set(ylabel = None)
sns.heatmap(total_data[["charges", "bmi"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = total_data, x = "children", y = "charges").set(ylabel = None)
sns.heatmap(total_data[["charges", "children"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

# Adjust layout / Ajustar el diseño
plt.tight_layout()

# Show the plot / Mostrar el gráfico
plt.show()

fig, axis = plt.subplots(figsize = (7, 5))

sns.countplot(data = total_data, x = "smoker", hue = "region")

# Show the plot / Mostrar el gráfico
plt.show()

# Factorize the categorical variables
total_data["sex"] = pd.factorize(total_data["sex"])[0]
total_data["smoker"] = pd.factorize(total_data["smoker"])[0]
total_data["region"] = pd.factorize(total_data["region"])[0]

fig, axes = plt.subplots(figsize=(10, 8))

sns.heatmap(total_data[["age", "sex", "bmi", "children", "smoker", "region", "charges"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()

sns.pairplot(data = total_data)

total_data.describe()

fig, axes = plt.subplots(2, 3, figsize = (15, 10))

sns.boxplot(ax = axes[0, 0], data = total_data, y = "age")
sns.boxplot(ax = axes[0, 1], data = total_data, y = "bmi")
sns.boxplot(ax = axes[0, 2], data = total_data, y = "children")
sns.boxplot(ax = axes[1, 0], data = total_data, y = "charges")
sns.boxplot(ax = axes[1, 1], data = total_data, y = "smoker")
sns.boxplot(ax = axes[1, 2], data = total_data, y = "sex")

plt.tight_layout()

plt.show()

# Stats for charges / Estadísticas para charges
charges_stats = total_data["charges"].describe()
charges_stats

# IQR for charges / Rango intercuartílico para charges

charges_iqr = charges_stats["75%"] - charges_stats["25%"]
upper_limit = charges_stats["75%"] + 1.5 * charges_iqr
lower_limit = charges_stats["25%"] - 1.5 * charges_iqr


# Clean the outliers / Limpiar los outliers

total_data = total_data[total_data["charges"] > 0]

bmi_stats = total_data["bmi"].describe()
bmi_stats

# IQR for bmi / Rango intercuartílico para bmi
bmi_iqr = bmi_stats["75%"] - bmi_stats["25%"]

upper_limit = bmi_stats["75%"] + 1.5 * bmi_iqr
lower_limit = bmi_stats["25%"] - 1.5 * bmi_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(bmi_iqr, 2)}")

# Clean the outliers / Limpiar los outliers

total_data = total_data[total_data["bmi"] <= upper_limit]


# Stats for age / Estadísticas para age

age_stats = total_data["age"].describe()
age_stats


# IQR for age / Rango intercuartílico para age

age_iqr = age_stats["75%"] - age_stats["25%"]

upper_limit = age_stats["75%"] + 1.5 * age_iqr
lower_limit = age_stats["25%"] - 1.5 * age_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(age_iqr, 2)}")

# Stats for children / Estadísticas para children

children_stats = total_data["children"].describe()
children_stats

# IQR for children / Rango intercuartílico para children

children_iqr = children_stats["75%"] - children_stats["25%"]

upper_limit = children_stats["75%"] + 1.5 * children_iqr
lower_limit = children_stats["25%"] - 1.5 * children_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(children_iqr, 2)}")

# Count NaN values / Contar valores NaN
total_data.isnull().sum().sort_values(ascending = False)


from sklearn.preprocessing import MinMaxScaler

num_variables = ["age", "bmi", "children", "sex", "smoker", "region"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(total_data[num_variables])
df_scal = pd.DataFrame(scal_features, index = total_data.index, columns = num_variables)
df_scal["charges"] = total_data["charges"]
df_scal.head()

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scal.drop("charges", axis=1)
y = df_scal["charges"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# chi2 needs: (1) non-negative X, (2) categorical y
scaler = MinMaxScaler()
X_train_pos = scaler.fit_transform(X_train)
X_test_pos = scaler.transform(X_test)
y_train_bins = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
selection_model = SelectKBest(chi2, k=4)
selection_model.fit(X_train_pos, y_train_bins)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(
    selection_model.transform(X_train_pos), columns=X_train.columns.values[ix]
)
X_test_sel = pd.DataFrame(
    selection_model.transform(X_test_pos), columns=X_train.columns.values[ix]
)
X_train_sel.head()


X_train_sel["charges"] = list(y_train)
X_test_sel["charges"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)

# Load train and test datasets / Cargar los conjuntos de datos de entrenamiento y prueba
train_data = pd.read_csv("../data/processed/clean_train.csv")
test_data = pd.read_csv("../data/processed/clean_test.csv") 

train_data.head()


# Visualization of features vs charges / Visualización de características vs charges
fig, axis = plt.subplots(4, 2, figsize = (10, 16))
total_data = pd.concat([train_data, test_data])

sns.regplot(ax = axis[0, 0], data = total_data, x = "age", y = "charges")
sns.heatmap(total_data[["charges", "age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = total_data, x = "smoker", y = "charges")
sns.heatmap(total_data[["charges", "smoker"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1], cbar = False)

sns.regplot(ax = axis[2, 0], data = total_data, x = "sex", y = "charges")
sns.heatmap(total_data[["charges", "sex"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0], cbar = False)

sns.regplot(ax = axis[2, 1], data = total_data, x = "children", y = "charges")
sns.heatmap(total_data[["charges", "children"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 1], cbar = False)

plt.tight_layout()
plt.show()



X_train = train_data.drop(["charges"], axis = 1)
y_train = train_data["charges"]
X_test = test_data.drop(["charges"], axis = 1)
y_test = test_data["charges"]


# Model training / Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)


# Model parameters / Parámetros del modelo

print(f"Intercepto (a): {model.intercept_}")
for feature, coef in zip(X_train.columns, model.coef_):
    print(f"Coeficiente {feature}: {coef}")


    # Predictions / Predicciones

y_pred = model.predict(X_test)
y_pred

# Model evaluation / Evaluación del modelo

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")



