from utils import db_connect
engine = db_connect()

# your code here
from utils import db_connect
engine = db_connect()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

