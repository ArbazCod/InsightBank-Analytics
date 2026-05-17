from dotenv import load_dotenv
import os

load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML & Statistics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway



st.set_page_config(
    page_title="InsightBank Analytics | Enterprise Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .header-container {
        background: linear-gradient(-45deg, #1a237e, #283593, #0d47a1, #1565c0);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(26, 35, 126, 0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #e3f2fd;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    .kpi-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        border-left: 4px solid #1a237e;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.25);
    }
    
    .kpi-label {
        color: #546e7a;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        background: linear-gradient(135deg, #1a237e, #283593);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    
    .kpi-change {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .kpi-change.positive {
        background: rgba(40, 167, 69, 0.1);
        color: #28a745;
    }
    
    .kpi-change.negative {
        background: rgba(220, 53, 69, 0.1);
        color: #dc3545;
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a237e;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e9ecef;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .live-indicator {
        width: 12px;
        height: 12px;
        background: #28a745;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(26, 35, 126, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING FROM CSV
# =========================================================

@st.cache_data
def load_full_dataset():
    return pd.read_csv("bank_analytics_clean.csv")

df = load_full_dataset()

metrics = pd.DataFrame({
    "total_records": [len(df)],
    "avg_balance": [df["balance"].mean()],
    "avg_age": [df["age"].mean()]
})

# =========================================================
# SIDEBAR - DEFINE ALL VARIABLES HERE FIRST
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span class="live-indicator"></span>
        <span style="font-weight: 600; color: #1a237e;">Live Data Pipeline</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Advanced Filters")
    
    # Define all filter variables with default values
    selected_jobs = st.multiselect(
        "Job Categories",
        options=sorted(df['job'].unique()),
        default=sorted(df['job'].unique()),
        help="Select job categories to analyze"
    )
    
    selected_education = st.multiselect(
        "Education Level",
        options=sorted(df['education'].unique()),
        default=sorted(df['education'].unique()),
        help="Filter by education level"
    )
    
    selected_marital = st.multiselect(
        "Marital Status",
        options=sorted(df['marital'].unique()),
        default=sorted(df['marital'].unique()),
        help="Filter by marital status"
    )
    
    selected_housing = st.multiselect(
        "Housing Loan",
        options=sorted(df['housing'].unique()),
        default=sorted(df['housing'].unique()),
        help="Filter by housing loan status"
    )
    
    selected_loan = st.multiselect(
        "Personal Loan",
        options=sorted(df['loan'].unique()),
        default=sorted(df['loan'].unique()),
        help="Filter by personal loan status"
    )
    
    # Age range slider
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    age_range = st.slider(
        "Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )
    
    # Balance range slider
    max_balance = int(df['balance'].max())
    balance_range = st.slider(
        "Balance Range ($)",
        min_value=0,
        max_value=max_balance,
        value=(0, max_balance),
        step=1000,
        format="$%d"
    )
    
    # Campaign filters
    st.markdown("### 📞 Campaign Filters")
    
    max_campaign = int(df['campaign'].max())
    campaign_range = st.slider(
        "Campaign Contacts",
        min_value=1,
        max_value=max_campaign,
        value=(1, min(10, max_campaign)),
        step=1
    )
    
    max_duration = int(df['duration'].max())
    duration_range = st.slider(
        "Call Duration (seconds)",
        min_value=0,
        max_value=max_duration,
        value=(0, min(1000, max_duration)),
        step=30
    )
    
    # Previous contacts filter
    max_previous = int(df['previous'].max())
    previous_range = st.slider(
        "Previous Contacts",
        min_value=0,
        max_value=max_previous,
        value=(0, max_previous),
        step=1
    )
    
    st.markdown("---")
    
    # Apply filters button
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄 Apply Filters", type="primary", width='stretch'):
            st.success("✅ Filters applied successfully!")
    with col2:
        if st.button("🔄 Reset", width='stretch'):
            st.rerun()

# =========================================================
# APPLY FILTERS TO DATAFRAME
# =========================================================

# Create filtered dataframe (this runs every time, using sidebar variables)
filtered_df = df[
    (df['job'].isin(selected_jobs)) &
    (df['education'].isin(selected_education)) &
    (df['marital'].isin(selected_marital)) &
    (df['housing'].isin(selected_housing)) &
    (df['loan'].isin(selected_loan)) &
    (df['age'].between(age_range[0], age_range[1])) &
    (df['balance'].between(balance_range[0], balance_range[1])) &
    (df['campaign'].between(campaign_range[0], campaign_range[1])) &
    (df['duration'].between(duration_range[0], duration_range[1])) &
    (df['previous'].between(previous_range[0], previous_range[1]))
]

# Show warning if no data after filtering
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
    st.stop()

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header-container">
    <div class="header-title">🏦 InsightBank Analytics</div>
    <div class="header-subtitle">
        Enterprise-Grade Banking Intelligence Platform · AI-Powered Insights · Real-Time Analytics
    </div>
    <div style="margin-top: 1rem;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 20px; color: white; font-size: 0.9rem;">
            🟢 System Operational · Records: {:,} · Updated: {}
        </span>
    </div>
</div>
""".format(len(filtered_df), datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

# =========================================================
# KPI DASHBOARD
# =========================================================

st.markdown('<div class="section-header">📊 Executive Command Center</div>', unsafe_allow_html=True)

# Calculate KPIs
conv_rate = (filtered_df['y'] == 'yes').mean() * 100
avg_balance = filtered_df['balance'].mean()
avg_duration = filtered_df['duration'].mean()
avg_age = filtered_df['age'].mean()
contact_rate = (filtered_df['contact'] == 'cellular').mean() * 100
successful_conv = (filtered_df['y'] == 'yes').sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">👥 Total Customers</div>
        <div class="kpi-value">{len(filtered_df):,}</div>
        <div class="kpi-change positive">Filtered Dataset</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">✅ Successful Conversions</div>
        <div class="kpi-value">{successful_conv:,}</div>
        <div class="kpi-change positive">Rate: {conv_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">💰 Average Balance</div>
        <div class="kpi-value">${avg_balance:,.0f}</div>
        <div class="kpi-change positive">Age: {avg_age:.0f} yrs</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">📱 Mobile Contact Rate</div>
        <div class="kpi-value">{contact_rate:.1f}%</div>
        <div class="kpi-change positive">Duration: {avg_duration:.0f}s</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# JOB CATEGORY ANALYSIS
# =========================================================

st.markdown('<div class="section-header">🎯 Job Category Performance Analysis</div>', unsafe_allow_html=True)

# Job analysis
job_analysis = filtered_df.groupby('job').agg(
    Total_Customers=('y', 'count'),
    Conversions=('y', lambda x: (x == 'yes').sum()),
    Conversion_Rate=('y', lambda x: (x == 'yes').mean() * 100),
    Avg_Balance=('balance', 'mean'),
    Avg_Duration=('duration', 'mean')
).reset_index()

job_analysis = job_analysis.sort_values('Conversion_Rate', ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    # Bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=job_analysis['job'],
        y=job_analysis['Conversion_Rate'],
        text=[f"{rate:.1f}%" for rate in job_analysis['Conversion_Rate']],
        textposition='outside',
        marker=dict(
            color=job_analysis['Conversion_Rate'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Conversion %")
        ),
        hovertemplate="<b>%{x}</b><br>Conversion: %{y:.1f}%<br>Customers: %{customdata:,}<extra></extra>",
        customdata=job_analysis['Total_Customers']
    ))
    
    avg_rate = job_analysis['Conversion_Rate'].mean()
    fig.add_hline(
        y=avg_rate,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_rate:.1f}%",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Conversion Rate by Job Category",
        xaxis_title="Job Category",
        yaxis_title="Conversion Rate (%)",
        template="plotly_white",
        height=450,
        xaxis=dict(tickangle=-45)
    )
    
    st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown("### 📊 Top Performers")
    top_jobs = job_analysis.head(5)
    for _, row in top_jobs.iterrows():
        st.markdown(f"""
        <div style="background: white; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #1a237e;">{row['job']}</span>
                <span style="color: #28a745; font-weight: 700;">{row['Conversion_Rate']:.1f}%</span>
            </div>
            <div style="font-size: 0.8rem; color: #6c757d;">
                {row['Total_Customers']:,} customers · ${row['Avg_Balance']:,.0f} avg balance
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# CUSTOMER SEGMENTATION HEATMAP
# =========================================================

st.markdown('<div class="section-header">🔥 Customer Segmentation Matrix</div>', unsafe_allow_html=True)

# Create segmentation heatmap
segmentation = filtered_df.groupby(['age_group', 'balance_category']).agg(
    Total=('y', 'count'),
    Conversions=('y', lambda x: (x == 'yes').sum()),
    Conversion_Rate=('y', lambda x: (x == 'yes').mean() * 100)
).reset_index()

# Pivot for heatmap
heatmap_data = segmentation.pivot(
    index='age_group',
    columns='balance_category',
    values='Conversion_Rate'
).fillna(0)

# Heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='RdYlGn',
    text=[[f"{val:.1f}%" for val in row] for row in heatmap_data.values],
    texttemplate="%{text}",
    textfont={"size": 12},
    colorbar=dict(title="Conversion Rate %"),
    hovertemplate="Age: %{y}<br>Balance: %{x}<br>Conversion: %{z:.1f}%<extra></extra>"
))

fig_heatmap.update_layout(
    title="Conversion Rate by Age Group & Balance Category",
    xaxis_title="Balance Category",
    yaxis_title="Age Group",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_heatmap, width='stretch')

# =========================================================
# MACHINE LEARNING PREDICTIONS
# =========================================================

st.markdown('<div class="section-header">🤖 AI-Powered Predictive Analytics</div>', unsafe_allow_html=True)

if len(filtered_df) >= 100:
    try:
        # Prepare features
        feature_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']
        
        # Encode
        df_encoded = filtered_df.copy()
        label_encoders = {}
        
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            df_encoded[col] = label_encoders[col].fit_transform(df_encoded[col].astype(str))
        
        X = df_encoded[feature_cols + categorical_cols]
        y = (df_encoded['y'] == 'yes').astype(int)
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols + categorical_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance chart
            fig_imp = go.Figure(go.Bar(
                y=importance_df['Feature'][:10],
                x=importance_df['Importance'][:10],
                orientation='h',
                marker=dict(color=importance_df['Importance'][:10], colorscale='Viridis'),
                text=[f"{x:.3f}" for x in importance_df['Importance'][:10]],
                textposition='outside'
            ))
            
            fig_imp.update_layout(
                title="Top 10 Predictive Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_imp, width='stretch')
        
        with col2:
            # Model metrics
            cv_scores = cross_val_score(model, X, y, cv=5)
            roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
            
            st.markdown("### 📊 Model Performance")
            st.metric("Cross-Validation Score", f"{cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
            st.metric("Training Samples", f"{len(X):,}")
            
            # Prediction distribution
            probas = model.predict_proba(X)[:, 1]
            
            fig_dist = go.Figure(go.Histogram(
                x=probas, nbinsx=30,
                marker_color='#1a237e', opacity=0.7
            ))
            
            fig_dist.update_layout(
                title="Prediction Probability Distribution",
                xaxis_title="Conversion Probability",
                yaxis_title="Count",
                template="plotly_white",
                height=300
            )
            
            st.plotly_chart(fig_dist, width='stretch')
            
    except Exception as e:
        st.warning(f"⚠️ ML model training issue: {str(e)}")
else:
    st.info(f"📊 Need at least 100 records for ML analysis. Current: {len(filtered_df):,} records")

# =========================================================
# STATISTICAL ANALYSIS
# =========================================================

st.markdown('<div class="section-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔬 Conversion by Education Level")
    
    edu_analysis = filtered_df.groupby('education').agg(
        Count=('y', 'count'),
        Conversion_Rate=('y', lambda x: (x == 'yes').mean() * 100)
    ).reset_index()
    
    fig_edu = go.Figure()
    
    fig_edu.add_trace(go.Bar(
        x=edu_analysis['education'],
        y=edu_analysis['Conversion_Rate'],
        text=[f"{rate:.1f}%" for rate in edu_analysis['Conversion_Rate']],
        textposition='outside',
        marker_color='#1a237e'
    ))
    
    fig_edu.update_layout(
        title="Conversion Rate by Education",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_edu, width='stretch')

with col2:
    st.markdown("### 📊 Balance Distribution by Marital Status")
    
    fig_box = go.Figure()
    
    for status in filtered_df['marital'].unique():
        data = filtered_df[filtered_df['marital'] == status]['balance']
        fig_box.add_trace(go.Box(y=data, name=status, boxmean=True))
    
    fig_box.update_layout(
        title="Balance Distribution by Marital Status",
        yaxis_title="Balance ($)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_box, width='stretch')

# =========================================================
# CORRELATION ANALYSIS
# =========================================================

st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)

numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
corr_matrix = filtered_df[numeric_cols].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='RdBu',
    zmid=0,
    text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
    texttemplate="%{text}",
    textfont={"size": 12}
))

fig_corr.update_layout(
    title="Correlation Matrix of Numeric Features",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_corr, width='stretch')

# =========================================================
# CAMPAIGN ANALYSIS
# =========================================================

st.markdown('<div class="section-header">📞 Campaign Performance Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Contact type analysis
    contact_analysis = filtered_df.groupby('contact').agg(
        Count=('y', 'count'),
        Conversion_Rate=('y', lambda x: (x == 'yes').mean() * 100)
    ).reset_index()
    
    fig_contact = go.Figure(data=[
        go.Bar(name='Count', x=contact_analysis['contact'], y=contact_analysis['Count'],
               marker_color='#1a237e', yaxis='y'),
        go.Scatter(name='Conversion Rate', x=contact_analysis['contact'], 
                   y=contact_analysis['Conversion_Rate'],
                   mode='lines+markers', marker=dict(size=12, color='red'),
                   yaxis='y2')
    ])
    
    fig_contact.update_layout(
        title="Contact Type Performance",
        template="plotly_white",
        height=400,
        yaxis=dict(title="Count"),
        yaxis2=dict(title="Conversion Rate (%)", overlaying='y', side='right')
    )
    
    st.plotly_chart(fig_contact, width='stretch')

with col2:
    # Previous outcome analysis
    poutcome_analysis = filtered_df.groupby('poutcome').agg(
        Count=('y', 'count'),
        Conversion_Rate=('y', lambda x: (x == 'yes').mean() * 100)
    ).reset_index()
    
    fig_pout = go.Figure(go.Bar(
        x=poutcome_analysis['poutcome'],
        y=poutcome_analysis['Conversion_Rate'],
        text=[f"{rate:.1f}%" for rate in poutcome_analysis['Conversion_Rate']],
        textposition='outside',
        marker=dict(
            color=poutcome_analysis['Conversion_Rate'],
            colorscale='RdYlGn'
        )
    ))
    
    fig_pout.update_layout(
        title="Conversion by Previous Campaign Outcome",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_pout, width='stretch')

# =========================================================
# DATA EXPORT SECTION
# =========================================================

st.markdown('<div class="section-header">📋 Data Export & Reports</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    st.dataframe(
        filtered_df[numeric_cols].describe(),
        width='stretch'
    )

with col2:
    # Download filtered data
    st.markdown("### 💾 Download Data")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"banking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width='stretch'
    )
    
    # Sample data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(filtered_df.head(10), width='stretch')

with col3:
    # Quick insights
    st.markdown("### 💡 Quick Insights")
    
    insights = []
    
    if conv_rate > 15:
        insights.append(f"✅ High conversion rate: {conv_rate:.1f}%")
    else:
        insights.append(f"📊 Moderate conversion rate: {conv_rate:.1f}%")
    
    if avg_duration > 300:
        insights.append(f"⏱️ Long average calls: {avg_duration:.0f}s")
    
    if contact_rate > 60:
        insights.append(f"📱 Strong mobile contact: {contact_rate:.1f}%")
    
    for insight in insights:
        st.info(insight)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
">
    <h3 style="margin-bottom: 1rem;">🏦 InsightBank Analytics Platform</h3>
    <p style="color: #e0e0e0;">
        Enterprise-Grade Banking Intelligence · AI-Powered Analytics · Real-Time Insights
    </p>
    <p style="color: #b0b0b0; font-size: 0.9rem; margin-top: 1rem;">
        Data Points Analyzed: {len(filtered_df):,} · Filters Applied: {
            len(selected_jobs) + len(selected_education) + len(selected_marital) + 
            len(selected_housing) + len(selected_loan)
        } · Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
    <p style="color: #909090; font-size: 0.8rem;">
        © 2025 InsightBank · All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)