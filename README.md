# 🚢 TITANIC1 - Inteligencia a Bordo: Dashboard Titanic & Visión por Computadora

Dashboard interactivo basado en **Streamlit** que implementa metodología **CRISP-ML(Q)** para predicción de supervivencia del Titanic, análisis de costos y simulación de visión por computadora.

## 📋 Requisitos

- Python 3.11+
- Dependencias: `streamlit`, `pandas`, `scikit-learn`, `pillow`, `numpy`

## 🚀 Inicio Rápido

### 1️⃣ Entrenar los Modelos (Paso Inicial)

Antes de ejecutar la aplicación, debes generar los modelos entrenados:

```bash
cd /workspaces/TITANIC1
python train_models.py
```

Este script generará tres archivos pickle:
- `model_classifier.pkl` - Modelo Logistic Regression (predicción supervivencia)
- `model_regressor.pkl` - Modelo Linear Regression (predicción tarifa)
- `model_metadata.pkl` - Metadatos de entrenamiento

**Salida esperada:**
```
✅ ¡Modelos entrenados y guardados exitosamente!
```

### 2️⃣ Ejecutar la Aplicación Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📊 Características

### 🔮 Modelado Predictivo e Inferencia
- **Clasificación (Logistic Regression)**: Predice probabilidad de supervivencia
- **Regresión (Linear Regression)**: Predice costo del pasaje basado en atributos
- **Bucle de Retroalimentación**: Modelo bidireccional que garantiza consistencia

### 👁️ Visión por Computadora (Simulada)
- Detección y conteo facial en imágenes de embarque
- Bounding boxes automáticos
- Soporte para imágenes personalizadas (JPG/PNG)

### 🎨 Diseño Premium
- Tema oscuro con glassmorphism
- Interfaz responsive
- Animaciones suaves

## 📁 Estructura del Proyecto

```
.
├── app.py                      # App principal Streamlit
├── train_models.py             # Script de entrenamiento
├── titanic.csv                 # Dataset Titanic
├── titanic_analysis.ipynb      # Análisis exploratorio
├── assets/                     # Imágenes demostrativas
│   └── passenger_detector_ar.png
└── model_*.pkl                 # Modelos entrenados (generados)
```

## 📈 Métricas de Rendimiento

**Clasificador (Supervivencia):**
- Precisión: ~80%
- Muestras: 891 pasajeros

**Regresor (Tarifa):**
- R² Score: ~0.39
- Rango de tarifas: $0 - $512.33

## 🔧 Solución de Problemas

### Error: "No se encontraron los modelos entrenados"
1. Ejecuta `python train_models.py`
2. Verifica que los archivos `model_*.pkl` existan
3. Reinicia la aplicación: `streamlit run app.py`

### Error: ModuleNotFoundError
Instala las dependencias:
```bash
pip install streamlit pandas scikit-learn pillow numpy
```

## 📚 Metodología CRISP-ML(Q)

1. **Business Understanding** → Predicción supervivencia Titanic
2. **Data Understanding** → Análisis exploratorio del dataset
3. **Data Preparation** → Limpieza, imputación, codificación
4. **Modeling** → Clasificación + Regresión
5. **Evaluation** → Validación y métricas
6. **Monitoring** → Dashboard en tiempo real

## 👤 Autor

- **Proyecto**: TITANIC1
- **Repositorio**: https://github.com/yudyalmanzar32-source/TITANIC1