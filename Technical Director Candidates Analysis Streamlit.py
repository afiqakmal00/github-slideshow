import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from math import pi

# Load the data
@st.cache_data
def load_data():
    file_path = "C:/Users/USER/Downloads/Technical_Director_Candidates_Analysis.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Clean and preprocess data
def preprocess_data(df):
    # Convert percentage strings to numeric values
    for col in df.columns:
        if '%' in col and 'Formation' not in col:  # Skip formation columns
            # Remove % and convert to float
            df[col] = df[col].astype(str).str.rstrip('%').astype('float') / 100.0
    
    # Fill NA values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

df = preprocess_data(df)

# Calculate percentile ranks for all metrics
def calculate_percentiles(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    percentile_df = df.copy()
    
    for col in numeric_cols:
        # Higher is better for most metrics, but not all
        if 'GA' in col or 'xGA' in col or 'injuries' in col or 'Average minute of first sub' in col:
            percentile_df[col] = 1 - percentile_df[col].rank(pct=True)
        else:
            percentile_df[col] = percentile_df[col].rank(pct=True)
    
    return percentile_df

percentile_df = calculate_percentiles(df)

# Calculate category scores
def calculate_category_scores(percentile_df):
    category_scores = percentile_df.copy()
    
    # Define category columns - ensure we only include numeric columns
    numeric_cols = percentile_df.select_dtypes(include=[np.number]).columns
    
    tactical_cols = [col for col in numeric_cols if 'Tactical' in col]
    squad_cols = [col for col in numeric_cols if 'Squad Management' in col]
    transfer_cols = [col for col in numeric_cols if 'Transfer Strategy' in col]
    
    # Calculate average scores only for numeric columns
    category_scores['Tactical Score'] = percentile_df[tactical_cols].mean(axis=1)
    category_scores['Squad Management Score'] = percentile_df[squad_cols].mean(axis=1)
    category_scores['Transfer Strategy Score'] = percentile_df[transfer_cols].mean(axis=1)
    category_scores['Overall Score'] = percentile_df[numeric_cols].mean(axis=1)
    
    return category_scores

category_scores = calculate_category_scores(percentile_df)

# Streamlit app
st.set_page_config(layout="wide", page_title="Technical Director Candidate Analysis")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #f0f2f6;
    }
    .header {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #2c3e50 !important;
    }
    .subheader {
        font-size: 18px !important;
        color: #34495e !important;
    }
    .metric-box {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .radar-chart {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("⚽ Technical Director Candidate Analysis")
st.markdown("""
    Compare and evaluate technical director candidates based on their tactical, squad management, 
    and transfer strategy performance metrics. The scoring system uses percentile ranking across all available metrics.
    """)

# Sidebar filters
st.sidebar.header("Filters & Options")
selected_candidates = st.sidebar.multiselect(
    "Select Candidates to Compare",
    options=df['Candidate'].unique(),
    default=df['Candidate'].tolist()
)

# Category selection
st.sidebar.subheader("Comparison Categories")
compare_tactical = st.sidebar.checkbox("Tactical Metrics", value=True)
compare_squad = st.sidebar.checkbox("Squad Management Metrics", value=True)
compare_transfer = st.sidebar.checkbox("Transfer Strategy Metrics", value=True)

# Filter data based on selections
filtered_df = df[df['Candidate'].isin(selected_candidates)]
filtered_percentiles = percentile_df[percentile_df['Candidate'].isin(selected_candidates)]
filtered_scores = category_scores[category_scores['Candidate'].isin(selected_candidates)]

# Main content
if not selected_candidates:
    st.warning("Please select at least one candidate to compare.")
else:
    # Overall scores
    st.header("Overall Candidate Scores")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Overall score bar chart
        fig = px.bar(
            filtered_scores.sort_values('Overall Score', ascending=False),
            x='Candidate',
            y='Overall Score',
            color='Overall Score',
            color_continuous_scale='Viridis',
            text='Overall Score',
            title="Overall Performance Score (Percentile Rank)"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(yaxis_range=[0,1], uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score breakdown
        st.markdown("### Score Breakdown")
        for _, row in filtered_scores.iterrows():
            with st.expander(f"{row['Candidate']} - {row['Club & Season']}"):
                st.metric("Overall Score", f"{row['Overall Score']:.2f}")
                st.metric("Tactical Score", f"{row['Tactical Score']:.2f}")
                st.metric("Squad Management", f"{row['Squad Management Score']:.2f}")
                st.metric("Transfer Strategy", f"{row['Transfer Strategy Score']:.2f}")
    
    # Radar chart for category comparison
    # Radar chart for category comparison
if compare_tactical or compare_squad or compare_transfer:
    st.header("Category Comparison Radar Chart")
    
    categories = []
    if compare_tactical:
        categories.append('Tactical Score')
    if compare_squad:
        categories.append('Squad Management Score')
    if compare_transfer:
        categories.append('Transfer Strategy Score')
    
    if len(categories) > 1:
        # Prepare data for radar chart - manual approach
        fig = go.Figure()
        
        for candidate in filtered_scores['Candidate'].unique():
            candidate_data = filtered_scores[filtered_scores['Candidate'] == candidate]
            
            # Close the loop by repeating the first value
            values = candidate_data[categories].values.flatten().tolist()
            values += values[:1]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],  # Close the loop
                fill='toself',
                name=candidate
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]  # Percentile scores range from 0-1
                )),
            showlegend=True,
            title="Category Comparison (Percentile Scores)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least two categories for radar chart comparison.")
    
    # Detailed metric comparison
    st.header("Detailed Metric Comparison")
    
    # Select metrics to compare
    all_metrics = []
    if compare_tactical:
        all_metrics.extend([col for col in df.columns if 'Tactical' in col])
    if compare_squad:
        all_metrics.extend([col for col in df.columns if 'Squad Management' in col])
    if compare_transfer:
        all_metrics.extend([col for col in df.columns if 'Transfer Strategy' in col])
    
    if not all_metrics:
        st.warning("Please select at least one category to compare metrics.")
    else:
        # Let user select metrics to display
        selected_metrics = st.multiselect(
            "Select specific metrics to compare",
            options=all_metrics,
            default=all_metrics[:5]  # Show first 5 by default
        )
        
        if selected_metrics:
            # Create tabs for different visualization options
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Heatmap", "Parallel Coordinates"])
            
            with tab1:
                # Bar chart comparison
                st.subheader("Metric Comparison (Percentile Rank)")
                
                # Create a column for each metric
                cols = st.columns(len(selected_metrics))
                
                for i, metric in enumerate(selected_metrics):
                    with cols[i]:
                        st.markdown(f"**{metric}**")
                        fig = px.bar(
                            filtered_percentiles,
                            x='Candidate',
                            y=metric,
                            color='Candidate',
                            text=metric
                        )
                        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                        fig.update_layout(yaxis_range=[0,1], showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Heatmap
                st.subheader("Performance Heatmap")
                
                heatmap_data = filtered_percentiles.set_index('Candidate')[selected_metrics]
                
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Metric", y="Candidate", color="Percentile"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Viridis',
                    aspect="auto"
                )
                fig.update_xaxes(side="top")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Parallel coordinates
                st.subheader("Parallel Coordinates Plot")
                
                # Normalize the selected metrics for better visualization
                normalized_data = filtered_df.copy()
                for metric in selected_metrics:
                    if metric in normalized_data.columns:
                        normalized_data[metric] = MinMaxScaler().fit_transform(
                            normalized_data[[metric]]
                        )
                
                fig = px.parallel_coordinates(
                    normalized_data,
                    dimensions=['Candidate'] + selected_metrics,
                    color='Candidate',
                    title="Parallel Coordinates Comparison (Normalized Metrics)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
   
    
   
#Path: cd "C:/Users/USER/Desktop/Rembatz Analisis/Python Script"
# Execute: streamlit run "Technical Director Candidates Analysis Streamlit.py"
