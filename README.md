# 🩺 26. Disease Prediction from Symptoms

> Predict potential diseases based on patient-reported symptoms using Machine Learning.

This is a Machine Learning web app that predicts diseases like Fungal infection, Allergy, Drug Reaction, and Common Cold based on symptoms.

**Live Demo:** https://disease-prediction-app-f2qmzjdpd6oeg5jxh6k5zq.streamlit.app/

### ✨ Key Features
- **Patient-Reported Symptoms (UI):** User-friendly checkboxes for symptoms (Itching, Skin Rash, Sneezing, etc.)
- **Multi-Disease Prediction:** Predicts multiple diseases using XGBoost Classifier
- **Confidence Scores:** Shows prediction confidence using `predict_proba`
- **Doctor Recommendations:** Recommends consulting a Dermatologist or General Physician

### 🛠️ Tech Stack Used
- **Python**
- **Scikit-learn** (LabelEncoder)
- **XGBoost** (XGBClassifier)
- **Streamlit** (For Web UI)
- **Pandas & Numpy**

### 🚀 How to Run Locally
1. Clone the repo:
   `git clone https://github.com/MShehroze-Stack/disease-prediction-app.git`
2. Install requirements:
   `pip install -r requirements.txt`
3. Run the app:
   `streamlit run app.py`

## 📸 Screenshots

### Home Page - Symptom Selection
![Home Page](Screenshot_20251026_151413.png)

### Prediction Result
![Prediction Result](Screenshot_20251026_151501.png)

### 👨‍💻 Author
**M Shehroze Baig** - [LinkedIn](https://www.linkedin.com/in/m-shehroze-baig/)
