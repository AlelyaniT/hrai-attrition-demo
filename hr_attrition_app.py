# hr_attrition_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ========== Sample HR Dataset ==========
data = {
    'Employee_ID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Department': ['Sales', 'Engineering', 'HR', 'Sales', 'Engineering', 'HR', 'Sales', 'Engineering'],
    'Tenure': [2, 4, 1, 5, 3, 2, 4, 1],
    'Absences': [8, 2, 12, 4, 5, 9, 3, 7],
    'Performance': [7, 9, 6, 8, 7, 5, 8, 6],
    'Sentiment': ['frustrated', 'happy', 'stressed', 'neutral', 'happy', 'frustrated', 'neutral', 'stressed'],
    'Left_Company': [1, 0, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Sentiment mapping
sentiment_map = {'frustrated': -1, 'stressed': -0.5, 'neutral': 0, 'happy': 1}
df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map)

# Model setup
X = df[['Tenure', 'Absences', 'Performance', 'Sentiment_Score']]
y = df['Left_Company']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ========== Streamlit Interface ==========
st.set_page_config(page_title="HR Attrition Predictor", layout="centered")

st.title("🔍 HR Attrition Risk Prediction Demo")
st.subheader("Ministry of Human Resources – Predictive Analytics")

# Input form
with st.form("attrition_form"):
    st.markdown("### 📋 Enter Employee Details")
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    absences = st.slider("Absences (Days)", 0, 30, 5)
    performance = st.slider("Performance (1-10)", 1, 10, 7)
    sentiment = st.selectbox("Sentiment", options=list(sentiment_map.keys()))
    submitted = st.form_submit_button("Predict Risk")

# Predict
if submitted:
    sentiment_score = sentiment_map[sentiment]
    new_data = [[tenure, absences, performance, sentiment_score]]
    risk = model.predict_proba(new_data)[0][1] * 100
    st.success(f"🔴 Predicted Attrition Risk: **{risk:.1f}%**")

# Show dataset
with st.expander("📊 View Sample Data"):
    st.dataframe(df)

# Charts
st.markdown("### 📈 Attrition Analysis")

# Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig1, ax1 = plt.subplots()
feat_imp.plot(kind='barh', color='skyblue', ax=ax1)
ax1.set_title("Top Factors Driving Attrition")
st.pyplot(fig1)

# Absences vs Attrition
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='Absences', y='Left_Company', hue='Department', ax=ax2, s=100)
ax2.set_title("Absences vs. Attrition")
st.pyplot(fig2)
