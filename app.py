import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spam Detector Pro", layout="centered")

st.title("📊 Email Spam Classifier Dashboard")
st.caption("Logistic Regression model to detect and filter out spam messages.")
st.write("---")

# 1. Load Model and Vectorizer safely
try:
    with open('spam_model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('vectorizer.pkl', 'rb') as v_file:
        vectorizer = pickle.load(v_file)
except FileNotFoundError:
    st.error("🚨 Model files missing! Please upload 'spam_model.pkl' and 'vectorizer.pkl' to your GitHub repository.")

# 2. Dataset Overview Section (Optional EDA view)
st.subheader("📈 Dataset Insight")
if st.checkbox("Show EDA Graphs"):
    try:
        df_view = pd.read_csv('spam.csv', encoding='latin-1').dropna(how="any", axis=1)
        df_view.columns = ['Label', 'Email_Text']
        df_view['Length'] = df_view['Email_Text'].apply(len)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.countplot(x='Label', data=df_view, ax=ax[0])
        ax[0].set_title("Class Distribution")
        
        sns.histplot(x='Length', hue='Label', data=df_view, bins=30, ax=ax[1])
        ax[1].set_xlim(0, 250)
        ax[1].set_title("Email Length Distribution")
        st.pyplot(fig)
    except:
        st.warning("Make sure 'spam.csv' is uploaded in GitHub to see graphs.")

st.write("---")

# 3. User Input and Prediction Area
st.subheader("✉️ Live Email Spam Tester")
user_email = st.text_area("Paste the email text you want to check below:", placeholder="Type or paste text here...")

if st.button("🔍 Check Email"):
    if user_email.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Step 1: Text transform numbers me convert karna
        transformed_text = vectorizer.transform([user_email])
        
        # Step 2: Prediction lena
        prediction = model.predict(transformed_text)
        
        # Step 3: Result dikhana
        st.write("### Prediction Result:")
        if prediction[0] == 1:
            st.error("🚨 ALERT: This email is classified as **SPAM**!")
        else:
            st.success("✅ Safe: This email is classified as **HAM** (Legitimate).")
