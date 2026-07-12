import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Spam Sentinel Pro",
    page_icon="✉️",
    layout="centered"
)

# --- Custom CSS Styling for Premium UI ---
st.markdown("""
    <style>
    /* Button Customization */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    /* Input Text Area Rounded Corners */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
    }
    /* Checkbox styling */
    .stCheckbox {
        background-color: #1e293b;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title and Subtitle
st.title("✉️ Email Spam Sentinel")
st.caption("A smart Machine Learning system powered by Logistic Regression to secure your inbox.")
st.write("---")

# 1. Load Model and Vectorizer safely
try:
    with open('spam_model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('vectorizer.pkl', 'rb') as v_file:
        vectorizer = pickle.load(v_file)
except FileNotFoundError:
    st.error("🚨 Model files missing! Please upload 'spam_model.pkl' and 'vectorizer.pkl' to your GitHub repository.")

# 2. Dataset Overview Section (EDA view with modern styling)
st.subheader("📊 Dataset Insights")
if st.checkbox("🔍 Click to view Dataset Graphs"):
    try:
        df_view = pd.read_csv('spam.csv', encoding='latin-1').dropna(how="any", axis=1)
        df_view.columns = ['Label', 'Email_Text']
        df_view['Length'] = df_view['Email_Text'].apply(len)
        
        # Modern Dark Styled Graphs
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#0f172a')
        
        # Left Graph
        sns.countplot(x='Label', data=df_view, ax=ax[0], palette=['#3b82f6', '#ef4444'])
        ax[0].set_title("Class Distribution", fontsize=12, fontweight='bold')
        ax[0].set_facecolor('#1e293b')
        
        # Right Graph
        sns.histplot(x='Length', hue='Label', data=df_view, bins=30, ax=ax[1], palette=['#3b82f6', '#ef4444'])
        ax[1].set_xlim(0, 250)
        ax[1].set_title("Email Length Distribution", fontsize=12, fontweight='bold')
        ax[1].set_facecolor('#1e293b')
        
        st.pyplot(fig)
    except:
        st.warning("Make sure 'spam.csv' is uploaded in GitHub to see graphs.")

st.write("---")

# 3. User Input and Prediction Area
st.subheader("🔮 Smart Scan")
st.write("Paste the suspicious email content below to analyze it instantly:")
user_email = st.text_area("", placeholder="Enter email content here...", height=150)

if st.button("🚀 Analyze Message"):
    if user_email.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Vectorize input
        transformed_text = vectorizer.transform([user_email])
        
        # Predict
        prediction = model.predict(transformed_text)
        
        st.write("### Diagnostics Result:")
        if prediction[0] == 1:
            st.error("🚨 **CRITICAL ALERT:** This message contains patterns highly correlated with **SPAM** activities.")
        else:
            st.success("✅ **SAFE REGION:** This message appears legitimate and safe for your inbox (**HAM**).")
