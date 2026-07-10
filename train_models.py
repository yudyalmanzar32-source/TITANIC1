"""
Script para entrenar y guardar los modelos del Titanic
Genera: model_classifier.pkl, model_regressor.pkl, model_metadata.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

def train_models():
    """Entrena los modelos de clasificación y regresión."""
    
    # Cargar datos
    csv_path = 'titanic.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró {csv_path}")
    
    print(f"📂 Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # Limpieza y preparación de datos
    print("\n🔧 Limpiando datos...")
    
    # Imputación de valores faltantes
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    print(f"  • Edad mediana: {age_median:.2f}")
    
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f"  • Puerto de embarque más frecuente: {embarked_mode}")
    
    fare_median = df['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_median)
    print(f"  • Tarifa mediana: {fare_median:.2f}")
    
    # Codificación de variables categóricas
    print("\n📊 Codificando variables categóricas...")
    df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked_encoded'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Verificar ausencia de valores NaN después de la codificación
    if df[['Sex_encoded', 'Embarked_encoded']].isnull().any().any():
        print("⚠️ Advertencia: Valores NaN después de la codificación")
        df['Sex_encoded'] = df['Sex_encoded'].fillna(0)
        df['Embarked_encoded'] = df['Embarked_encoded'].fillna(2)
    
    print(f"  ✓ Variables categóricas codificadas")
    
    # Preparar características para clasificación (Supervivencia)
    print("\n🤖 Entrenando Clasificador (Logistic Regression)...")
    X_clf = df[['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']].values
    y_clf = df['Survived'].values
    
    print(f"  • Muestras de entrenamiento: {X_clf.shape[0]}")
    print(f"  • Características: {X_clf.shape[1]}")
    print(f"  • Clase 0 (No sobrevivió): {(y_clf == 0).sum()}")
    print(f"  • Clase 1 (Sobrevivió): {(y_clf == 1).sum()}")
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_clf, y_clf)
    clf_score = clf.score(X_clf, y_clf)
    print(f"  ✓ Clasificador entrenado (Precisión: {clf_score:.4f})")
    
    # Preparar características para regresión (Tarifa)
    print("\n📈 Entrenando Regresor (Linear Regression)...")
    X_reg = df[['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Embarked_encoded', 'Survived']].values
    y_reg = df['Fare'].values
    
    print(f"  • Muestras de entrenamiento: {X_reg.shape[0]}")
    print(f"  • Características: {X_reg.shape[1]}")
    print(f"  • Rango de tarifas: ${y_reg.min():.2f} - ${y_reg.max():.2f}")
    
    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    reg_score = reg.score(X_reg, y_reg)
    print(f"  ✓ Regresor entrenado (R² Score: {reg_score:.4f})")
    
    # Metadatos
    meta = {
        'age_median': float(age_median),
        'fare_median': float(fare_median),
        'embarked_mode': embarked_mode,
        'classifier_accuracy': float(clf_score),
        'regressor_r2': float(reg_score),
        'training_samples': int(X_clf.shape[0]),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    # Guardar modelos
    print("\n💾 Guardando modelos...")
    
    with open('model_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print(f"  ✓ model_classifier.pkl guardado")
    
    with open('model_regressor.pkl', 'wb') as f:
        pickle.dump(reg, f)
    print(f"  ✓ model_regressor.pkl guardado")
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print(f"  ✓ model_metadata.pkl guardado")
    
    print("\n✅ ¡Modelos entrenados y guardados exitosamente!")
    print(f"\n📊 Resumen de Metadatos:")
    for key, value in meta.items():
        print(f"  • {key}: {value}")
    
    return clf, reg, meta

if __name__ == "__main__":
    try:
        train_models()
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
