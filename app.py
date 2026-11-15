import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Upwork Job Market Analytics & ML Recommendation Engine",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Enterprise Job Market ML Prediction & Recommendation Engine")
st.caption("Automated Scikit-Learn Salary Estimator Pipeline & Personalized Job Recommendation Engine")

# Load artifacts
@st.cache_resource
def load_models():
    pipeline = joblib.load('model_pipeline.joblib')
    rec_data = joblib.load('rec_engine.joblib')
    return pipeline, rec_data

pipeline, rec_data = load_models()
rec_tfidf = rec_data['tfidf']
rec_matrix = rec_data['matrix']
rec_df = rec_data['data']

tab1, tab2 = st.tabs([
    "🤖 Automated Salary Estimator (Pipeline)",
    "🎯 Personalized Recommendation Engine"
])

with tab1:
    st.header("Automated Scikit-Learn Pipeline: Salary Estimator")
    st.markdown("Pass raw job titles and employer countries directly through the compiled `sklearn.pipeline.Pipeline`.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        test_title = st.text_input("Enter Job Title / Requirement:", value="Senior Machine Learning Engineer PyTorch")
    with col_b:
        test_country = st.selectbox(
            "Select Client Country:",
            ["United States", "United Kingdom", "Canada", "Australia", "Germany", "India", "Unknown"]
        )
        
    if st.button("Predict Expected Hourly Rate", type="primary"):
        input_df = pd.DataFrame([{'title': test_title, 'country': test_country}])
        predicted_rate = pipeline.predict(input_df)[0]
        adjusted_rate = max(3.0, predicted_rate)
        
        st.success(f"### Estimated Hourly Compensation: **${adjusted_rate:.2f} / hr**")
        st.info("Pipeline applies TF-IDF n-gram vectorization and categorical one-hot encoding on raw inputs.")

with tab2:
    st.header("Personalized Job Recommendation Engine")
    st.markdown("Match your skill profile against 240,000+ job postings using TF-IDF and Cosine Similarity.")
    
    user_query = st.text_input("Enter your skills or target role:", value="Full Stack React Python developer")
    
    col_c, col_d = st.columns(2)
    with col_c:
        top_k = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)
    with col_d:
        min_filter_rate = st.number_input("Minimum rate / budget ($):", min_value=0, value=25)
        
    if st.button("Find Matching Jobs", type="primary"):
        query_vec = rec_tfidf.transform([user_query])
        sims = cosine_similarity(query_vec, rec_matrix).flatten()
        top_indices = sims.argsort()[::-1]
        
        recs = []
        for idx in top_indices:
            if sims[idx] <= 0:
                break
            row = rec_df.iloc[idx]
            val = row['hourly_mid'] if row['is_hourly'] else row['budget']
            if pd.notna(val) and val < min_filter_rate:
                continue
                
            recs.append({
                "Job Title": row['title'],
                "Match Score": f"{sims[idx]:.1%}",
                "Contract": "Hourly" if row['is_hourly'] else "Fixed Budget",
                "Rate / Budget": f"${row['hourly_mid']:.1f}/hr" if row['is_hourly'] else f"${row['budget']:.0f}",
                "Client Country": row['country'],
                "Link": row['link']
            })
            if len(recs) >= top_k:
                break
                
        if recs:
            st.dataframe(pd.DataFrame(recs), use_container_width=True)
        else:
            st.warning("No postings found matching criteria.")
