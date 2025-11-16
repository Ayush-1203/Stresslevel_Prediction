# Stresslevel_Prediction

---

# 🧠 Stress Level Prediction & Wellness Analyzer

A Machine Learning–based web application built using **Streamlit** that predicts a user’s **stress level** (Low, Moderate, High) using 20 psychological, lifestyle, academic, and health features.
The app includes a **login system**, **prediction history**, a **trained ML model**, and a **modern UI** with supportive wellness tips.

---

## 📌 Project Features

* ✔️ Stress prediction using a trained Random Forest Model
* ✔️ 20 input features covering emotional, academic & lifestyle factors
* ✔️ Beautiful UI with custom CSS
* ✔️ Login & Signup system
* ✔️ Saves results in `history.csv`
* ✔️ Stores users in `users.csv`
* ✔️ Encouraging mental-wellness advice based on prediction

---

## 📁 Folder Structure

```
Stress-Prediction-App/
│── app.py
│── best_model.pkl
│── scaler.pkl
│── users.csv
│── history.csv
│── requirements.txt
│── README.md
│── Stress_Level_Prediction_Final_Submission.ipynb  (optional)
```

---

## 📊 Model Performance

| Model                           | Accuracy |
| ------------------------------- | -------- |
| Logistic Regression             | 78%      |
| SVM                             | 88%      |
| Gradient Boosting               | 89%      |
| **Random Forest (Final Model)** | **91%**  |

---

## 🔧 Technologies Used

* Python
* Streamlit
* Pandas, NumPy
* Scikit-learn
* Joblib

---

## ▶️ How to Run the Project

### 1️⃣ Install the requirements

```
pip install -r requirements.txt
```

### 2️⃣ Run the web app

```
streamlit run app.py
```

### 3️⃣ App opens at:

```
http://localhost:8501
```

---

## 🧪 How the Prediction Works

1. User enters 20 stress-related features
2. Data is scaled using `scaler.pkl`
3. Model (`best_model.pkl`) predicts stress level
4. Result + wellness message is shown
5. Prediction is saved to `history.csv`

---

## 📬 Developer

**Ayush Kumar,Ananya Jain**
Stress Prediction Project – 2025

---

### ✅ Copy this entire text and paste it inside your GitHub README file.
