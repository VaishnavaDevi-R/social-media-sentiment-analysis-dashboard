# 🏗️ System Architecture

## 🔁 Pipeline Flow

Text Data  
↓  
Text Cleaning & Preprocessing  
↓  
TF-IDF Vectorization  
↓  
Random Forest Model  
↓  
Sentiment Prediction  
↓  
Dashboard Visualization  

---

## 📦 Components

### 1. Data Layer
- Raw data: `data/raw/`
- Processed data: `data/processed/`

### 2. Processing Layer
- Text cleaning
- Stopword removal
- Lowercasing

### 3. Feature Engineering
- TF-IDF vectorization

### 4. Model Layer
- Random Forest classifier

### 5. Application Layer
- Streamlit dashboard

### 6. Output Layer
- Predictions
- Charts
- Insights