import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Page Configuration & Styling ---
st.set_page_config(page_title="Tourism Analytics Pro", page_icon="✈️", layout="wide")

# Optional: A touch of custom CSS to make the title pop
st.markdown("""
    <style>
    .title-text { color: #1E3A8A; font-weight: 700; margin-bottom: 0px;}
    .subtitle-text { color: #6B7280; font-size: 1.1rem; margin-top: -10px; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. Load Models and Data ---
@st.cache_resource
def load_artifacts():
    reg_model = joblib.load('models/regression_model.pkl')
    clf_model = joblib.load('models/classification_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    df = pd.read_csv('data/cleaned_tourism_data.csv')
    return reg_model, clf_model, encoders, df

reg_model, clf_model, encoders, df = load_artifacts()

# --- 3. Main Header ---
st.markdown('<h1 class="title-text">🌍 Tourism Experience Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">AI-Powered Insights & Personalized Travel Recommendations</p>', unsafe_allow_html=True)
st.divider()

# --- 4. Sidebar: Professional User Input Form ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2060/2060284.png", width=80) # Adds a nice travel icon
st.sidebar.title("Traveler Profile")
st.sidebar.markdown("Configure the parameters below to generate insights.")

with st.sidebar.expander("📍 Location Details", expanded=True):
    continent = st.selectbox("Continent", encoders['Continent'].classes_)
    country = st.selectbox("Country", encoders['Country'].classes_)
    region = st.selectbox("Region", encoders['Region'].classes_)

with st.sidebar.expander("⭐ Preferences & Timing", expanded=True):
    attraction_type = st.selectbox("Preferred Attraction Type", encoders['AttractionType'].classes_)
    visit_month = st.slider("Visit Month", min_value=1, max_value=12, value=6, format="Month %d")

st.sidebar.divider()
st.sidebar.caption("Powered by Machine Learning")

# --- 5. Main Dashboard Layout (Tabs) ---
tab1, tab2, tab3 = st.tabs(["🔮 Behavior Prediction", "🗺️ Smart Recommendations", "📊 Global Insights"])

# --- TAB 1: Classification Task ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Travel Companion AI")
        st.write("Our classification model analyzes global trends to predict your most likely travel group based on your profile.")
        predict_btn = st.button("Generate Prediction", type="primary", use_container_width=True)
        
    with col2:
        if predict_btn:
            # Convert text inputs to encoded numbers
            cont_enc = encoders['Continent'].transform([continent])[0]
            count_enc = encoders['Country'].transform([country])[0]
            reg_enc = encoders['Region'].transform([region])[0]
            attr_enc = encoders['AttractionType'].transform([attraction_type])[0]
            
            input_data = pd.DataFrame([[cont_enc, count_enc, reg_enc, attr_enc, visit_month]], 
                                      columns=['Continent_Encoded', 'Country_Encoded', 'Region_Encoded', 'AttractionType_Encoded', 'VisitMonth'])
            
            with st.spinner('Analyzing patterns...'):
                pred_enc = clf_model.predict(input_data)[0]
                pred_mode = encoders['VisitMode'].inverse_transform([pred_enc])[0]
            
            st.success("Prediction Complete!")
            st.metric(label="Predicted Visit Mode", value=f"👥 {pred_mode}")
            st.info(f"**Insight:** Travelers from **{country}** visiting **{attraction_type}** locations in Month **{visit_month}** predominantly travel as **{pred_mode}**.")

# --- TAB 2: Recommendation System ---
with tab2:
    st.subheader(f"Top Destinations for: {attraction_type}")
    
    # Process Recommendations
    recommendations = df[df['AttractionType'] == attraction_type]
    top_recs = recommendations.groupby(['Attraction', 'AttractionCityName', 'Country'])['Rating'].mean().reset_index()
    top_recs = top_recs.sort_values(by='Rating', ascending=False).head(5)
    
    if not top_recs.empty:
        # Highlight the #1 Recommendation
        best_match = top_recs.iloc[0]
        st.markdown(f"### 🏆 Top Match: **{best_match['Attraction']}**")
        st.markdown(f"📍 *{best_match['AttractionCityName']}, {best_match['Country']}* |  ⭐ **{best_match['Rating']:.1f}/5.0**")
        st.divider()
        
        # Display the rest in a clean dataframe
        st.write("### 📌 Other Highly Rated Options")
        top_recs.columns = ['Attraction Name', 'City', 'Country', 'Average User Rating']
        
        # Style the dataframe to hide the index and format the rating
        st.dataframe(top_recs.style.format({"Average User Rating": "{:.2f}"}), use_container_width=True, hide_index=True)
    else:
        st.warning("No highly-rated attractions found for this specific category in the current database.")

# --- TAB 3: Visualizations & KPIs ---
with tab3:
    st.subheader("Data-Driven Tourism Trends")
    
    # KPI Row
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Transactions Analyzed", f"{len(df):,}")
    kpi2.metric("Unique Attractions", f"{df['AttractionId'].nunique():,}")
    kpi3.metric("Global Countries Tracked", f"{df['Country'].nunique():,}")
    
    st.divider()
    
    # Charts Row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Top 5 Attraction Categories**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(y='AttractionType', data=df, order=df['AttractionType'].value_counts().index[:5], palette='Blues_r', ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel("Total Visits")
        sns.despine() # Makes the chart look cleaner
        st.pyplot(fig)
        
    with chart_col2:
        st.markdown("**User Demographics (By Continent)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Continent', data=df, palette='viridis', ax=ax, order=df['Continent'].value_counts().index)
        plt.xticks(rotation=45)
        ax.set_xlabel("")
        ax.set_ylabel("Total Users")
        sns.despine()
        st.pyplot(fig)