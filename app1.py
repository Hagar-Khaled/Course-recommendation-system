import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="🎓 Personalized Course Recommendation System",
    page_icon="📚",
    layout="wide"
)

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
    <style>
        /* App background */
        .stApp {
            background-color: #f7f9fc;
        }

        /* Title */
        .title {
            font-size: 36px !important;
            color: #2c3e50;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }

        /* Subtitle */
        .subtitle {
            font-size: 18px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }

        /* Course Card */
        .course-card {
            background-color: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }

        .course-card:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        }

        /* Buttons */
        .stButton>button {
            background-color: #3498db;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }

        .stButton>button:hover {
            background-color: #2980b9;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Load Preprocessed Data & Models
# =========================
@st.cache_resource
def load_data():
    data = joblib.load("courses_df1.pkl")
    model = joblib.load("embedding_model1.pkl")
    embeddings = joblib.load("embeddings1.pkl")
    return data, model, embeddings

data, model, embeddings = load_data()

# =========================
# App Title & Description
# =========================
st.markdown('<div class="title">🎓 Personalized Course Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find the best courses based on your interests 🚀</div>', unsafe_allow_html=True)

# =========================
# Search Box
# =========================
user_input = st.text_input(
    "🔍 Type a keyword or course name",
    placeholder="e.g. Machine Learning, Data Science, AI, Python ..."
)

# =========================
# Recommend Courses
# =========================
if st.button("🔎 Recommend Courses"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a keyword to search.")
    else:
        # Encode user input
        user_embedding = model.encode([user_input], convert_to_tensor=False)
        similarities = cosine_similarity([user_embedding], embeddings)[0]

        # Get top 5 recommendations
        top_indices = similarities.argsort()[::-1][:5]
        recommendations = data.iloc[top_indices].reset_index(drop=True)

        # Show results
        st.subheader("🎯 Top 5 Recommended Courses:")
        for i, row in recommendations.iterrows():
            st.markdown(f"""
                <div class="course-card">
                    <h3>📌 {row['Course Name']}</h3>
                    <p><b>📖 Description:</b> {row['Course Description']}</p>
                    <p><b>🎯 Skills:</b> {row['Skills']}</p>
                    <p><b>🏛️ University:</b> {row['University']}</p>
                    <p><b>📊 Difficulty:</b> {row['Difficulty Level']}</p>
                </div>
            """, unsafe_allow_html=True)

        # Download button
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv,
            file_name="recommended_courses.csv",
            mime="text/csv"
        )

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#7f8c8d;'>🚀 Built with ❤️ using Streamlit</div>",
    unsafe_allow_html=True
)
