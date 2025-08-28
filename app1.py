import streamlit as st
st.set_page_config(page_title="Course Recommendation System", layout="wide")
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load saved files
data = joblib.load("courses_df1.pkl")
model = SentenceTransformer('all-MiniLM-L6-v2')  # load directly instead of pickled model
embeddings = joblib.load("embeddings1.pkl")

st.set_page_config(page_title="🎓 Smart Course Recommendation System", page_icon="🎓", layout="centered")
st.title("🎓 Smart Personalized Course Recommendation System")
st.write("Get the **most relevant** courses based on your interests!")

# Input keyword
keyword = st.text_input("🔍 Type a topic or course name:")

def recommend_courses(query, top_n=6):
    # Encode user query into semantic embedding
    query_embedding = model.encode([query], convert_to_tensor=False)

    # Calculate similarity with all courses
    similarity_scores = cosine_similarity(query_embedding, embeddings)[0]

    # Get top matches
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = data.iloc[top_indices]
    return recommendations

if st.button("🔎 Get Recommendations"):
    if keyword.strip() == "":
        st.warning("⚠️ Please enter a topic or course name.")
    else:
        results = recommend_courses(keyword)
        st.subheader("📌 Recommended Courses:")
        for idx, row in results.iterrows():
            st.markdown(f"**{row['Course Name']}** — {row['University']} ({row['Difficulty Level']})")
            st.write(f"*Skills:* {row['Skills']}")
            st.write(f"*Description:* {row['Course Description']}")
            st.markdown("---")

