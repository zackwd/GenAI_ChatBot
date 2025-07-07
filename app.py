import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import shap

st.title("Traffic Fatality Risk Prediction Dashboard")

# Load prepared parquet data directly
df = pd.read_parquet("fars_2023_realistic_fatality_ready.parquet")
st.write("Data Preview:", df.head())

# Use INJ_SEV_BIN as target
df_clean = df.dropna()
X = df_clean.drop(columns=['INJ_SEV_BIN'])
y = df_clean['INJ_SEV_BIN']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# User input section
st.subheader("Try Your Own Prediction")
impact1 = st.number_input("Enter IMPACT1", min_value=0, max_value=100, value=0)
rollover = st.number_input("Enter ROLLOVER", min_value=0, max_value=100, value=0)
drinking = st.number_input("Enter DRINKING", min_value=0, max_value=100, value=0)

user_data = pd.DataFrame({
    'IMPACT1': [impact1],
    'ROLLOVER': [rollover],
    'DRINKING': [drinking]
})

if st.button("Predict Fatality Risk"):
    prediction = clf.predict(user_data)[0]
    prediction_proba = clf.predict_proba(user_data)[0][1]
    
    st.markdown(f"**Predicted Fatality Risk (0=No, 1=Yes): {prediction}**")

    # Large, colored, eye-catching display
    if prediction_proba > 0.5:
        prob_text = f"<div style='font-size:32px; color:red; font-weight:bold;'>Probability of Fatality: {prediction_proba:.2f}</div>"
    else:
        prob_text = f"<div style='font-size:28px; color:green; font-weight:bold;'>Probability of Fatality: {prediction_proba:.2f}</div>"

    st.markdown(prob_text, unsafe_allow_html=True)


# Model evaluation
st.subheader("Model Evaluation on Test Data")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

auc_score = roc_auc_score(y_test, y_proba)
st.write(f"ROC AUC Score: {auc_score:.4f}")

# SHAP Feature Importance
st.subheader("SHAP Feature Importance")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Using bar plot for clarity with few features
st.write("Feature importance based on SHAP values:")
fig_shap = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)

