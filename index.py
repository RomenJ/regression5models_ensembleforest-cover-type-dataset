import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import mean_squared_error as MSE, r2_score, mean_absolute_error as MAE
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import joblib
#dataset:https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset
SEED = 1

def mejor_modelo(modelos, X, y):
    """Evalúa varios modelos y devuelve el mejor según el MSE en la validación cruzada."""
    mejor_mse = float('inf')
    mejor_nombre = None
    
    for nombre, modelo in modelos.items():
        # Calcular MSE usando validación cruzada
        scores = -cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
        mse = scores.mean()
        
        # Actualizar el mejor modelo si es necesario
        if mse < mejor_mse:
            mejor_mse = mse
            mejor_nombre = nombre
    
    return mejor_nombre, mejor_mse

def load_data(filename):
    """Cargar el conjunto de datos desde el archivo CSV."""
    try:
        data = pd.read_csv(filename)
        print(f"Archivo '{filename}' cargado correctamente.")
        return data
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filename}'")
        return None
    except Exception as e:
        print(f"Error al cargar el archivo '{filename}': {e}")
        return None

def plot_predictions_vs_actual(y_test, y_pred, title):
    """Graficar las predicciones vs los valores reales"""
    sns.set_style("whitegrid")
    sns.scatterplot(x=y_test, y=y_pred)
    y_test = list(y_test)
    y_pred = list(y_pred)
    for i in range(len(y_test)):
        plt.plot([y_test[i], y_test[i]], [y_test[i], y_pred[i]], '-', color='blue', alpha=0.2)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title(title)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.savefig(f'Gráfico_de_predicciones_{title}.png')  # Guardar el gráfico
    plt.show()

def plot_distribution(y_test, y_pred, title):
    """Graficar la distribución de los valores reales y predichos"""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_test, color='blue', label='Valores Reales', shade=True)
    sns.kdeplot(y_pred, color='orange', label='Valores Predichos', shade=True)
    plt.title(title)
    plt.xlabel('Valores')
    plt.ylabel('Densidad')
    plt.legend()
    plt.savefig(f'Distribución_{title}.png')  # Guardar el gráfico
    plt.show()

def plot_bar_chart(model_scores, title):
    """Graficar un gráfico de barras para comparar el rendimiento de los modelos"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()))
    plt.title(title)
    plt.xlabel('Modelo')
    plt.ylabel('MSE promedio')
    plt.savefig(f'Bar_chart_{title}.png')  # Guardar el gráfico
    plt.show()

# Cargar el conjunto de datos desde el archivo CSV
def main():
    # for test :
    data = load_data('covtype.csv').head(1000)
    #use ONLY if you have time or a high computer capacity
    #data = load_data('covtype.csv')
    top_cor_labels=[]  
    if data is not None:
        print("Información del conjunto de datos:")
        print(data.info())
        print("\nPrimeras 5 filas del conjunto de datos:")
        print(data.head(5))
        print("\nForma del conjunto de datos:")
        print(data.shape)
        print("\nValores nulos en el conjunto de datos:")
        print(data.isna().sum())
        print("\nVariables relevantes:")

        # Calcular la matriz de correlación
        corr_matrix = data.corr()

        # Seleccionar las variables con mayor correlación
        top_corr_features = corr_matrix.index[abs(corr_matrix["Cover_Type"]) > 0.1]
        top_cor_labels.append(top_corr_features)
        
        # Crear un nuevo dataframe con las variables seleccionadas
        top_corr_data = data[top_corr_features]

        # Calcular la matriz de correlación para las variables seleccionadas
        top_corr_matrix = top_corr_data.corr()

        # Crear un mapa de calor
        plt.figure(figsize=(12, 8))
        sns.heatmap(top_corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Mapa de Calor de las Variables con Mayor Correlación")
        plt.savefig('Mapa_de_Calor_de_Variables.png')  # Guardar el gráfico
        plt.show()  

        # Definir las características (features) y la variable objetivo
        X = data[['Slope', 'Horizontal_Distance_To_Roadways',
                'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
                'Wilderness_Area4', 'Soil_Type2', 'Soil_Type6', 'Soil_Type10',
                'Soil_Type22', 'Soil_Type23', 'Soil_Type29', 'Soil_Type38',
                'Soil_Type39', 'Soil_Type40', 'Cover_Type']]
        y = data['Elevation']
        
        # Imprimir estadísticas de la variable objetivo
        print("\nEstadísticas de la variable objetivo 'Elevation':")
        print("Rango de valores:", y.min(), "-", y.max())
        print("Mediana:", y.median())
        print("Desviación estándar:", y.std())
        print("Percentiles:")
        print(y.describe(percentiles=[.25, .5, .75, .90, .95, .99]))

        # Dividir el conjunto de datos en datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

        # Modelos
        models = {
            "Gradient Boosting": GradientBoostingRegressor(max_depth=1, random_state=SEED, n_estimators=300),
            "Random Forest": RandomForestRegressor(random_state=SEED),
            "Extra Trees": ExtraTreesRegressor(random_state=SEED),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Linear Regression": LinearRegression()
        }

        model_scores = {}
        for name, model in models.items():
            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Predecir en los datos de prueba
            y_pred = model.predict(X_test)

            # Calcular el error cuadrático medio
            mse = MSE(y_test, y_pred)
            print(f"\n{name}:")
            print('MSE: {:.2f}'.format(mse))
            model_scores[name] = mse
            
            # Calcular el Root Mean Squared Error (RMSE)
            rmse = mse ** 0.5
            print('RMSE: {:.2f}'.format(rmse))
            
            # Calcular el coeficiente de determinación (R²)
            r2 = r2_score(y_test, y_pred)
            print('R²: {:.2f}'.format(r2))

            # Calcular el Error Absoluto Medio (MAE)
            mae = MAE(y_test, y_pred)
            print('MAE: {:.2f}'.format(mae))

            # Plot predictions vs actuals
            plot_predictions_vs_actual(y_test, y_pred, name)
            
            # Plot distribution of predicted and actual values
            plot_distribution(y_test, y_pred, name)

            # Curvas de aprendizaje
            train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            test_mean = test_scores.mean(axis=1)
            test_std = test_scores.std(axis=1)
            
            plt.figure()
            plt.plot(train_sizes, train_mean, 'o-', color="r", label="Entrenamiento")
            plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validación cruzada")
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
            plt.title(f"Curva de Aprendizaje - {name}")
            plt.xlabel("Tamaño del conjunto de entrenamiento")
            plt.ylabel("Puntuación")
            plt.legend(loc="best")
            plt.grid()
            plt.savefig(f'Curva_de_Aprendizaje_{name}.png')  # Guardar el gráfico
            plt.show()

        # Graficar un gráfico de barras para comparar el rendimiento de los modelos
        plot_bar_chart(model_scores, "Comparación de MSE promedio entre modelos")

        # Encuentra el mejor modelo con validación cruzada
        mejor_nombre, mejor_mse = mejor_modelo(models, X, y)
        print(f"\nEl mejor modelo tomando como mayor peso la validación cruzada sería: {mejor_nombre} con un MSE promedio de {mejor_mse}")
        print("Tenga en cuenta que esta recomenación le da mayor peso a la validación cruzada.")
        print("Análisis de R2, RMSE, MSE relevan que otros modelos pueden tener desempeños mejores.")
        print("https://datascientest.com/es/cross-validation-definicion-e-importancia")

if __name__ == "__main__":
    main()
