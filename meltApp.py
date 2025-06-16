import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io

# Set page configuration
st.set_page_config(layout="wide")

# Introduction and Problem Statement
st.title("CleanShield Albedo & Melt Risk Analysis")
st.markdown("""
### The Problem
Glaciers are melting due to rising temperatures, pollution (PM2.5), and reduced albedo, threatening water, ecosystems, and sea levels. CleanShield predicts melt risk and empowers communities with solutions.

### How to Use
1. **Data Source**: Select curated data, upload a CSV, or provide a URL.
2. **Explore**: Preview data and train the model.
3. **Visualize**: Analyze trends with multiple plots.
4. **Predict & Act**: Get risk predictions, alerts, and an action plan.

**Note**: Melt risk: >75 (High), >50 (Medium), ≤50 (Low).
""")

# Sidebar for settings
st.sidebar.header("Settings")
upload_option = st.sidebar.radio("Data Source", ["Use Curated Data", "Upload CSV", "URL"], help="Choose data: curated, upload, or URL to a CSV.")

# Data loading with error handling
@st.cache_data
def load_data(source):
    try:
        if upload_option == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Upload with columns: pm25, temperature, etc.")
            if uploaded_file is not None:
                return pd.read_csv(uploaded_file)
        elif upload_option == "URL":
            url = st.text_input("Enter CSV URL", help="Public URL to CSV (e.g., Google Drive link).")
            if url:
                return pd.read_csv(url)
        else:
            return pd.read_csv("curated_glacier_observations.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure the CSV has required columns.")
        return None

df = load_data(upload_option)
if df is not None and not all(col in df.columns for col in ['pm25', 'temperature', 'humidity', 'elevation', 'albedo', 'solar_radiation', 'wind_speed', 'snow_depth', 'precipitation', 'melt_risk']):
    st.error("Missing required columns. Check your CSV.")
    df = None

if df is not None:
    st.subheader("Data Preview")
    st.write("Preview of environmental factors and melt risk.", df.head())
    st.caption("Columns: pm25 (µg/m³), temperature (°C), humidity (%), elevation (m), albedo, solar_radiation (W/m²), wind_speed (m/s), snow_depth (cm), precipitation (mm), melt_risk (index).")

    # Model training and evaluation
    st.subheader("Model Analysis")
    st.write("Train a Random Forest model to predict melt risk.")
    if st.button("Train Model"):
        features = ['pm25', 'temperature', 'humidity', 'elevation', 'albedo', 
                   'solar_radiation', 'wind_speed', 'snow_depth', 'precipitation']
        X = df[features]
        y = df['melt_risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=250, max_depth=15, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"**Mean Squared Error**: {mse:.2f}")
        st.write(f"**R² Score**: {r2:.2f}")
        st.write("Model evaluates melt risk prediction accuracy.")

        feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        st.write("**Feature Importance**: Key melt risk drivers.")
        st.bar_chart(feature_importance.set_index('feature'))

    # Visualizations
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Heatmap")
        st.write("Shows relationships between factors and melt risk.")
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        st.pyplot(plt)

    with col2:
        st.subheader("Melt Risk Distribution")
        st.write("Displays melt risk spread.")
        plt.figure(figsize=(8, 6))
        sns.histplot(df['melt_risk'], bins=30, kde=True)
        st.pyplot(plt)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Temperature vs. Melt Risk")
        st.write("Scatter plot of temperature impact.")
        plt.figure(figsize=(8, 6))
        plt.scatter(df['temperature'], df['melt_risk'], alpha=0.5)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Melt Risk")
        st.pyplot(plt)

    with col4:
        st.subheader("Box Plot of Key Variables")
        st.write("Identifies outliers in albedo and PM2.5.")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df[['albedo', 'pm25']])
        plt.ylabel("Value")
        st.pyplot(plt)

    # Simulate time series if no date column
    st.subheader("Melt Risk Trend (Simulated)")
    st.write("Shows a simulated trend of melt risk over time.")
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(df)), df['melt_risk'], label="Melt Risk")
    plt.xlabel("Time (Simulated Index)")
    plt.ylabel("Melt Risk")
    plt.legend()
    st.pyplot(plt)

    # Prediction and Solution System
    st.subheader("Melt Risk Prediction & Solutions")
    if 'model' in locals():
        st.write("Adjust values to predict melt risk and get solutions.")
        input_data = {}
        for feature in features:
            input_data[feature] = st.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()), help=f"Slide {feature} value.")
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write(f"**Predicted Melt Risk**: {prediction:.2f}")

        # Alert and Solution System
        if prediction > 75:
            st.error("**High Melt Risk Alert (>75)**: Immediate action needed!")
            st.write("**Solutions**:")
            st.write("- Install shading to reduce solar exposure.")
            st.write("- Enforce PM2.5 pollution controls.")
            st.write("- Deploy artificial snowmaking.")
        elif prediction > 50:
            st.warning("**Medium Melt Risk Alert (>50)**: Preventive steps required.")
            st.write("**Solutions**:")
            st.write("- Monitor temperature and albedo.")
            st.write("- Reduce local pollution.")
            st.write("- Plan water management.")
        else:
            st.success("**Low Melt Risk Alert (≤50)**: Maintain vigilance.")
            st.write("**Solutions**:")
            st.write("- Continue monitoring.")
            st.write("- Promote sustainability.")
            st.write("- Educate communities.")

        # Comparative Analysis
        avg_melt_risk = df['melt_risk'].mean()
        st.write(f"**Comparison**: Prediction ({prediction:.2f}) vs. Average ({avg_melt_risk:.2f}).")
        if prediction > avg_melt_risk:
            st.write("**Insight**: Above average—prioritize mitigation.")
        else:
            st.write("**Insight**: Below average—focus on prevention.")

        # Real-Time Alert Simulation
        if prediction > 75:
            st.error("**Urgent Notification**: High melt risk detected! Act now.")

        # Dynamic Community Action Plan
        if st.button("Generate Action Plan"):
            latex_content = r"""
  \documentclass[a4paper,12pt]{article}
  \usepackage[utf8]{inputenc}
  \usepackage{geometry}
  \geometry{margin=1in}
  \usepackage{fancyhdr}
  \pagestyle{fancy}
  \fancyhf{}
  \rhead{CleanShield Action Plan}
  \lhead{Glacier Melt Mitigation}
  \cfoot{\thepage}

  \begin{document}

  \section*{Community Action Plan for Glacier Melt Mitigation}
  \subsection*{Risk Assessment}
  Predicted Melt Risk: """ + str(prediction) + r"""
  - Status: """ + ("High (>75)" if prediction > 75 else "Medium (>50)" if prediction > 50 else "Low (≤50)") + r"""

  \subsection*{Recommended Actions}
  Based on the risk level, consider:
  \begin{itemize}
      \item """ + ("Install shading structures." if prediction > 75 else "Monitor temperature and albedo." if prediction > 50 else "Continue regular monitoring.") + r"""
      \item """ + ("Enforce air pollution controls." if prediction > 75 else "Reduce local pollution sources." if prediction > 50 else "Promote sustainable practices.") + r"""
      \item """ + ("Deploy artificial snowmaking." if prediction > 75 else "Plan water management." if prediction > 50 else "Educate communities.") + r"""
  \end{itemize}

  \subsection*{Local Resources}
  - Collaborate with environmental groups.
  - Seek government funding for projects.
  - Engage community leaders.

  \end{document}
  """
            buffer = io.BytesIO()
            st.download_button(label="Download Action Plan (PDF)", data=buffer, file_name=f"action_plan_{prediction:.0f}.pdf", mime="application/pdf")
            st.write("Download this PDF to share with your community.")