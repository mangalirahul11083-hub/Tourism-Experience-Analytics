# 🌍 Tourism Experience Analytics: Classification, Prediction, and Recommendation System [cite: 1]

## 📌 Project Overview
[cite_start]Tourism agencies and travel platforms aim to enhance user experiences by leveraging data to provide personalized recommendations, predict user satisfaction, and classify potential user behavior[cite: 3]. [cite_start]This machine learning project analyzes a multi-relational dataset of over 52,000 user transactions, travel patterns, and attraction features to achieve three primary objectives: Regression, Classification, and Recommendation[cite: 4]. 

[cite_start]The final deliverable is an interactive Streamlit web dashboard[cite: 85, 199].

## 🎯 Key Objectives & Machine Learning Models
1. [cite_start]**Regression (Predicting Attraction Ratings):** Developed a Random Forest Regressor to predict the rating (1-5) a user might give to a tourist attraction based on historical data, user demographics, and attraction features [cite: 11-13].
2. [cite_start]**Classification (User Visit Mode Prediction):** Created a Random Forest Classifier to predict the mode of visit (e.g., Business, Family, Couples, Friends) based on user and attraction data [cite: 24-26].
3. [cite_start]**Recommendation (Personalized Suggestions):** Implemented a Content-Based filtering recommendation system to suggest unvisited tourist attractions tailored to a user's highly-rated historical preferences [cite: 37-39, 46].

## 💡 Business Insights & Value
* [cite_start]**Personalized Recommendations:** Suggests attractions based on past visits and demographics, improving the overall user experience and increasing customer retention[cite: 6, 9].
* [cite_start]**Tourism Analytics:** Provides dynamic visual insights into popular attractions and regions[cite: 7].
* [cite_start]**Customer Segmentation:** Classifies users into specific segments based on their travel behavior, enabling targeted marketing and promotions[cite: 8].

## 🛠️ Tech Stack & Skills Highlighted
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (RandomForestRegressor, RandomForestClassifier, LabelEncoder, MinMaxScaler)
* **Deployment:** Streamlit, Joblib
* [cite_start]**Core Skills:** Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, Predictive Modeling [cite: 115-120]

## 📁 Project Structure
```text
📦 Tourism Experience Analytics
 ┣ 📂 data
 ┃ ┣ 📜 cleaned_tourism_data.csv    # Preprocessed master dataset
 ┃ ┗ 📜 (Original 9 Excel files)
 ┣ 📂 models
 ┃ ┣ 📜 classification_model.pkl    # Trained Random Forest Classifier
 ┃ ┣ 📜 regression_model.pkl        # Trained Random Forest Regressor
 ┃ ┗ 📜 label_encoders.pkl          # Scikit-learn Label Encoders
 ┣ 📜 tourism_experience.ipynb      # Complete Jupyter Notebook (EDA & Training)
 ┣ 📜 app.py                        # Streamlit Web Application
 ┗ 📜 requirements.txt              # Project dependencies
