# ======================================================================================
# INTUIT DEMAND PLANNING STRATEGIC COMMAND CENTER
#
# A single-file Streamlit application for the Sr. Manager, Efficiency & Annual Planning.
#
# VERSION: Best-in-Class Strategic & AI-Integrated Edition (Unabridged)
#
# This dashboard provides a real-time, strategic view of Intuit's demand and supply
# ecosystem across the Consumer (TurboTax) and Small Business (QuickBooks) groups.
# It is designed to facilitate the end-to-end annual planning cycle, drive defect
# elimination, quantify efficiency savings, and lead the integration of AI into
# core planning and operational workflows.
#
# It integrates principles from:
#   - Lean Six Sigma & Root Cause Analysis (RCA)
#   - Statistical Process Control (SPC) for Service Operations
#   - Machine Learning for Time-Series Forecasting & Driver Analysis
#   - Strategic Program Management & Top-Down/Bottom-Up Financial Planning
#
# To Run:
# 1. Save this code as 'intuit_strategic_dashboard_final.py'
# 2. Create 'requirements.txt' with specified libraries.
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run from your terminal: streamlit run intuit_strategic_dashboard_final.py
#
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ======================================================================================
# SECTION 1: APP CONFIGURATION & STYLING
# ======================================================================================
st.set_page_config(page_title="Intuit Strategic Command Center", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main .block-container { padding: 1rem 3rem 3rem; }
    .stMetric { background-color: #fcfcfc; border: 1px solid #e0e0e0; border-left: 5px solid #0077C5; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #F0F2F6; border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #FFFFFF; box-shadow: 0 -2px 5px rgba(0,0,0,0.1); border-bottom-color: #FFFFFF !important; }
    .st-expander { border: 1px solid #E0E0E0 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# Define Intuit's Brand Colors
INTUIT_COLORS = {'blue': '#0077C5', 'green': '#00855A', 'orange': '#FF6B00', 'red': '#D92D20', 'grey': '#8D9096'}

# ======================================================================================
# SECTION 2: SME-DRIVEN DATA SIMULATION
# ======================================================================================
@st.cache_data(ttl=600)
def generate_master_data():
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=52, freq='W-SUN'))
    
    # AOP Data
    aop_data = {
        'Segment': ['TurboTax', 'QuickBooks'], 'Top_Down_Target_Users_M': [100, 12.5],
        'Bottom_Up_Plan_Users_M': [98.5, 12.3], 'AOP_Headcount_Plan': [5000, 7500],
        'AOP_Budget_M_USD': [1500, 2200]
    }
    aop_df = pd.DataFrame(aop_data)

    # Defect Data
    defects = ['Login Failure', 'Payment Error', 'QBO-Bank Sync', 'Tax Form Import', 'Mobile Crash']
    defect_data = []
    for week in dates:
        for defect in defects:
            base_vol = 5000 if defect == 'Login Failure' else 2000
            defect_data.append({'Week': week, 'Defect_Category': defect, 'Contact_Volume': np.random.randint(base_vol*0.8, base_vol*1.2) * (1 - (week - dates[0]).days / 500) })
    defect_df = pd.DataFrame(defect_data)
    defect_df['Cost_Impact_USD'] = defect_df['Contact_Volume'] * np.random.uniform(20, 40)

    # Forecast Model Data
    forecast_dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=104, freq='W-SUN'))
    forecast_df = pd.DataFrame({'Week': forecast_dates, 'Segment': ['TurboTax', 'QuickBooks'] * 52})
    forecast_df['Marketing_Spend_M_USD'] = np.random.uniform(5, 20, 104) * (1 + np.sin(np.arange(104) * (2 * np.pi / 52)) * 0.5)
    forecast_df['Is_Tax_Season'] = ((forecast_df['Week'].dt.month.isin([1,2,3,4]))).astype(int)
    forecast_df['New_Signups_K'] = (forecast_df['Marketing_Spend_M_USD']*10 + forecast_df['Is_Tax_Season']*50 + np.random.normal(0, 10, 104) + 50).clip(20)
    forecast_df['New_Signups_K'] = np.where(forecast_df['Segment'] == 'QuickBooks', forecast_df['New_Signups_K'] / 5, forecast_df['New_Signups_K'])

    # Marketing Channel Data
    channels = ['Paid Search', 'Social Media', 'Content Marketing', 'TV', 'Affiliates']
    marketing_data = {'Channel': channels, 'Spend_M_USD': [50, 25, 15, 75, 10], 'Acquisitions_K': [250, 150, 90, 300, 60]}
    marketing_df = pd.DataFrame(marketing_data)
    marketing_df['ROAS'] = (marketing_df['Acquisitions_K'] * 1000 * 150) / (marketing_df['Spend_M_USD'] * 1_000_000)

    # Customer Support Metrics
    support_metrics_data = {'Week': dates, 'Avg_Handle_Time_Sec': np.random.normal(480, 30, 52) * np.linspace(1, 0.85, 52), 'First_Contact_Resolution_Pct': np.random.normal(75, 5, 52) * np.linspace(1, 1.1, 52)}
    support_df = pd.DataFrame(support_metrics_data)

    # AI Opportunity Scoring
    ai_opp_data = {'Process': ['Contact Deflection', 'Fraud Detection', 'Churn Prediction', 'KB Article Generation', 'Marketing Budget Allocation', 'Tax Code Search'], 'Business_Unit': ['Shared', 'Shared', 'QuickBooks', 'Shared', 'Shared', 'TurboTax'], 'Impact_Score_100': [85, 95, 90, 70, 80, 92], 'Feasibility_Score_100': [90, 75, 70, 85, 60, 65]}
    ai_opp_df = pd.DataFrame(ai_opp_data)
    
    return aop_df, defect_df, forecast_df, marketing_df, support_df, ai_opp_df

# ======================================================================================
# SECTION 3: BEST-IN-CLASS VISUALIZATIONS & ML MODELS
# ======================================================================================
@st.cache_resource
def get_forecast_model(df):
    features = ['Marketing_Spend_M_USD', 'Is_Tax_Season']
    target = 'New_Signups_K'
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42); model.fit(X_train, y_train)
    importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    return model, X_test, y_test, importance

def plot_aop_reconciliation(df, segment):
    segment_data = df[df['Segment'] == segment]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Top-Down Target', x=['Users/Subs (M)'], y=segment_data['Top_Down_Target_Users_M'], text=segment_data['Top_Down_Target_Users_M'], textposition='auto', marker_color=INTUIT_COLORS['grey']))
    fig.add_trace(go.Bar(name='Bottom-Up Plan', x=['Users/Subs (M)'], y=segment_data['Bottom_Up_Plan_Users_M'], text=segment_data['Bottom_Up_Plan_Users_M'], textposition='auto', marker_color=INTUIT_COLORS['blue']))
    gap = segment_data['Top_Down_Target_Users_M'].iloc[0] - segment_data['Bottom_Up_Plan_Users_M'].iloc[0]
    fig.add_trace(go.Bar(name='Planning Gap', x=['Users/Subs (M)'], y=[gap], base=segment_data['Bottom_Up_Plan_Users_M'], text=f"Gap: {gap:,.1f}M", textposition='auto', marker_color=INTUIT_COLORS['red']))
    fig.update_layout(barmode='stack', title=f'<b>AOP Reconciliation: {segment}</b>', yaxis_title='Millions of Users/Subscribers')
    return fig

def plot_defect_spc_chart(df, defect_category):
    df_filtered = df[df['Defect_Category'] == defect_category]
    mean = df_filtered['Contact_Volume'].mean(); std_dev = df_filtered['Contact_Volume'].std(); ucl = mean + 3 * std_dev; lcl = max(0, mean - 3 * std_dev)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['Week'], y=df_filtered['Contact_Volume'], name='Weekly Volume', mode='lines+markers', line=dict(color=INTUIT_COLORS['blue'])))
    fig.add_hline(y=mean, line=dict(color='green', dash='dot'), name='Mean')
    fig.add_hline(y=ucl, line=dict(color='red', dash='dash'), name='UCL (+3σ)')
    fig.add_hline(y=lcl, line=dict(color='orange', dash='dash'), name='LCL (-3σ)')
    outliers = df_filtered[df_filtered['Contact_Volume'] > ucl]
    fig.add_trace(go.Scatter(x=outliers['Week'], y=outliers['Contact_Volume'], mode='markers', name='Special Cause Variation', marker=dict(symbol='x', color='red', size=12)))
    fig.update_layout(title=f'<b>SPC Chart for {defect_category} Defects</b>', yaxis_title='Weekly Contact Volume')
    return fig

def plot_marketing_mix_roas(df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Marketing Spend Allocation', 'Return on Ad Spend (ROAS) by Channel'), specs=[[{"type": "domain"}, {"type": "xy"}]])
    fig.add_trace(go.Pie(labels=df['Channel'], values=df['Spend_M_USD'], name="Spend", hole=.4, textinfo='label+percent'), 1, 1)
    fig.add_trace(go.Bar(y=df['Channel'], x=df['ROAS'], name="ROAS", orientation='h', marker_color=INTUIT_COLORS['green']), 1, 2)
    fig.update_yaxes(categoryorder='total ascending', row=1, col=2)
    fig.update_layout(title_text="<b>Marketing Intelligence: Channel Mix & Performance</b>", showlegend=False)
    return fig

def plot_support_kpi_dashboard(df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Avg. Handle Time (AHT)', 'First Contact Resolution (FCR)'))
    fig.add_trace(go.Indicator(mode="number+delta", value=df['Avg_Handle_Time_Sec'].iloc[-1], title="AHT (sec)", delta={'reference': df['Avg_Handle_Time_Sec'].mean(), 'decreasing': {'color': INTUIT_COLORS['green']}, 'increasing': {'color': INTUIT_COLORS['red']}}), 1, 1)
    fig.add_trace(go.Indicator(mode="number+delta", value=df['First_Contact_Resolution_Pct'].iloc[-1], title="FCR (%)", number={'suffix': '%'}, delta={'reference': df['First_Contact_Resolution_Pct'].mean()}), 1, 2)
    return fig
    
def plot_marketing_optimizer(df, budget):
    df['Adjusted_Spend_M'] = df['Spend_M_USD'] * (budget / df['Spend_M_USD'].sum())
    df['Adjusted_Acquisitions_K'] = (df['Adjusted_Spend_M'] * 1_000_000 * df['ROAS']) / 150 / 1000
    total_acq = df['Acquisitions_K'].sum()
    adjusted_acq = df['Adjusted_Acquisitions_K'].sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Original Plan', x=df['Channel'], y=df['Acquisitions_K']))
    fig.add_trace(go.Bar(name='Optimized Plan', x=df['Channel'], y=df['Adjusted_Acquisitions_K']))
    fig.update_layout(barmode='group', title='<b>Marketing Budget Scenario: Impact on Acquisitions</b>', yaxis_title='New Acquisitions (Thousands)')
    st.metric("Total Acquisitions (Optimized)", f"{adjusted_acq:,.0f}K", f"{adjusted_acq-total_acq:,.0f}K vs. Original Plan")
    return fig

def plot_ai_opportunity_matrix(df, segment_filter):
    if segment_filter != 'All Intuit':
        df = df[df['Business_Unit'].isin(['Shared', segment_filter])]
    avg_impact = df['Impact_Score_100'].mean()
    avg_feasibility = df['Feasibility_Score_100'].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Feasibility_Score_100'], y=df['Impact_Score_100'],
        mode='markers+text', text=df['Process'], textposition='top center',
        marker=dict(size=df['Impact_Score_100']/5, sizemin=10, color=df['Business_Unit'].astype('category').cat.codes, colorscale='viridis'),
        hovertext=df['Business_Unit'], name='AI Opportunities'
    ))
    fig.add_vline(x=avg_feasibility, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=avg_impact, line_width=1, line_dash="dash", line_color="grey")
    fig.update_layout(title='<b>AI Opportunity Prioritization Matrix</b>', xaxis_title='Feasibility Score (out of 100)', yaxis_title='Impact Score (out of 100)')
    fig.add_annotation(x=avg_feasibility*1.1, y=avg_impact*1.1, text="<b>🔥 QUICK WINS 🔥</b>", showarrow=False, font=dict(color=INTUIT_COLORS['green'], size=14))
    fig.add_annotation(x=avg_feasibility*0.9, y=avg_impact*1.1, text="<b>STRATEGIC BETS</b>", showarrow=False, font=dict(color=INTUIT_COLORS['blue']))
    fig.add_annotation(x=avg_feasibility*1.1, y=avg_impact*0.9, text="<b>INCREMENTAL</b>", showarrow=False, font=dict(color=INTUIT_COLORS['orange']))
    fig.add_annotation(x=avg_feasibility*0.9, y=avg_impact*0.9, text="<b>RE-EVALUATE</b>", showarrow=False, font=dict(color="grey"))
    return fig

# ======================================================================================
# SECTION 4: MAIN APPLICATION LAYOUT & SCIENTIFIC NARRATIVE
# ======================================================================================
st.title("🚀 Intuit Strategic Command Center")
st.markdown("##### Sr. Manager, Demand Planning: Efficiency & Annual Planning")
aop_df, defect_df, forecast_df, marketing_df, support_df, ai_opp_df = generate_master_data()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Intuit_logo.svg/1280px-Intuit_logo.svg.png", use_container_width=True)
st.sidebar.markdown("### Master Control")
segment_filter = st.sidebar.radio("Select Business Segment View:", ('All Intuit', 'TurboTax', 'QuickBooks'))

if segment_filter == 'All Intuit':
    filtered_forecast_df = forecast_df
else:
    filtered_forecast_df = forecast_df[forecast_df['Segment'] == segment_filter]

forecast_model, X_test, y_test, importance_df = get_forecast_model(filtered_forecast_df)

st.markdown("### I. Executive Command Center (Weekly Leadership View)")
st.subheader(f"Top-Line Strategic Health: {segment_filter}")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
aop_gap = (aop_df['Top_Down_Target_Users_M'].sum() - aop_df['Bottom_Up_Plan_Users_M'].sum()) / aop_df['Top_Down_Target_Users_M'].sum()
kpi_col1.metric("AOP Plan Gap", f"{aop_gap:.1%}", help="Gap between executive (Top-Down) targets and aggregated team (Bottom-Up) plans. Target < 1%.")
total_defect_cost = defect_df['Cost_Impact_USD'].sum() * 52 / 1_000_000
kpi_col2.metric("Annualized Defect Cost", f"${total_defect_cost:.1f}M", help="Total estimated annual cost of all tracked customer-facing defects across segments.")
mape = 100 * np.mean(np.abs(forecast_model.predict(X_test) - y_test) / y_test)
kpi_col3.metric("Strategic Forecast Accuracy (MAPE)", f"{mape:.1f}%", "-0.5% vs last quarter", "inverse")
fcr_trend = support_df['First_Contact_Resolution_Pct'].iloc[-1] - support_df['First_Contact_Resolution_Pct'].iloc[-4]
kpi_col4.metric("First Contact Resolution (FCR)", f"{support_df['First_Contact_Resolution_Pct'].iloc[-1]:.1f}%", f"{fcr_trend:+.1f} pts vs last month")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["**II. ANNUAL OPERATING PLAN (AOP) CENTER**", "**III. EFFICIENCY & DEFECT ELIMINATION**", "**IV. STRATEGIC FORECASTING & MARKETING**", "**V. AI-DRIVEN INNOVATION HUB**"])

with tab1:
    st.header("II. Annual Operating Plan (AOP) Center")
    st.markdown("_This section provides the end-to-end toolset to build, reconcile, and communicate the massive Intuit operating budget and plan._")
    st.subheader("A. AOP Reconciliation: Top-Down vs. Bottom-Up")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To visualize the critical phase of the annual planning cycle where executive-level targets (Top-Down) are compared against the sum of individual team and initiative plans (Bottom-Up).
        - **Method:** A grouped bar chart provides a direct visual comparison for each business segment. The gap between the bars represents the planning delta that must be closed through either new initiatives or target adjustments.
        - **Interpretation:** This chart is the primary tool for instilling an end-to-end planning cycle. A significant gap, as seen for TurboTax, triggers strategic discussions with senior leaders to identify new growth levers or efficiency savings to close the gap. It drives accountability by making the plan's core tension transparent.
        """)
    if segment_filter == 'All Intuit':
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(plot_aop_reconciliation(aop_df, 'TurboTax'), use_container_width=True)
        with col2: st.plotly_chart(plot_aop_reconciliation(aop_df, 'QuickBooks'), use_container_width=True)
    else:
        st.plotly_chart(plot_aop_reconciliation(aop_df, segment_filter), use_container_width=True)
    st.subheader("B. AOP Budget & Headcount Allocation")
    budget_fig = px.treemap(aop_df, path=['Segment'], values='AOP_Budget_M_USD', title='<b>AOP Budget Allocation by Segment</b>', color_discrete_map={'TurboTax':INTUIT_COLORS['green'], 'QuickBooks':INTUIT_COLORS['blue']})
    hc_fig = px.treemap(aop_df, path=['Segment'], values='AOP_Headcount_Plan', title='<b>AOP Headcount Allocation by Segment</b>', color_discrete_map={'TurboTax':INTUIT_COLORS['green'], 'QuickBooks':INTUIT_COLORS['blue']})
    col1, col2 = st.columns(2)
    col1.plotly_chart(budget_fig, use_container_width=True)
    col2.plotly_chart(hc_fig, use_container_width=True)

with tab2:
    st.header("III. Efficiency & Defect Elimination Engine")
    st.markdown("_This is the analytical core for identifying, prioritizing, and driving accountability for fixing customer-facing defects that create operational drag and erode margin._")
    st.subheader("A. Defect Reduction Monitoring (SPC)")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To determine if defect reduction efforts are producing a statistically significant improvement, or if weekly fluctuations are just random noise.
        - **Method:** A Shewhart control chart (SPC chart) plots the weekly contact volume for a specific defect. A center line represents the historical mean, and Upper/Lower Control Limits (UCL/LCL) are set at ±3 standard deviations (σ), defining the bounds of expected random variation.
        - **Interpretation:** A sustained downward trend or a point falling below the LCL is a statistical signal of a real process improvement. Conversely, a point above the UCL (a 'special cause' variation) indicates a process failure that requires immediate root cause analysis. This tool allows the leader to "drive accountability" with data, confirming whether a program leader's initiatives are truly impacting the numbers.
        """)
    defect_choice = st.selectbox("Select a Defect Category to Analyze:", defect_df['Defect_Category'].unique())
    st.plotly_chart(plot_defect_spc_chart(defect_df, defect_choice), use_container_width=True)
    st.subheader("B. Customer Support Efficiency Dashboard")
    st.plotly_chart(plot_support_kpi_dashboard(support_df), use_container_width=True)
    
with tab3:
    st.header("IV. Strategic Forecasting & Marketing Intelligence")
    st.markdown("_This section provides thought partnership to business leaders by moving beyond a single forecast to an interactive planning tool, enabling data-driven decisions on resource allocation and marketing optimization._")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("A. Marketing Channel Mix & ROAS")
        with st.expander("View Methodological Summary", expanded=True):
            st.markdown("""
            - **Purpose:** To provide a clear, data-driven view of marketing investment allocation and its effectiveness, directly supporting the optimization of marketing campaigns.
            - **Method:** A pie chart shows the allocation of the marketing budget across major channels. A horizontal bar chart ranks those same channels by their Return on Ad Spend (ROAS), a critical measure of efficiency.
            - **Interpretation:** This view enables powerful data storytelling. It can highlight misalignments, such as the largest slice of the budget (TV) having one of the lowest ROAS. This insight provides a strong, quantitative recommendation to senior leaders: reallocate budget from lower-performing channels like TV to higher-performing ones like Paid Search to maximize new user acquisition for the same overall spend.
            """)
        st.plotly_chart(plot_marketing_mix_roas(marketing_df), use_container_width=True)
    with col2:
        st.subheader("B. Marketing Budget Optimizer")
        st.markdown("Simulate budget reallocation to maximize acquisitions.")
        total_budget = marketing_df['Spend_M_USD'].sum()
        optimized_budget = st.slider("Total Marketing Budget (M USD)", float(total_budget*0.8), float(total_budget*1.2), float(total_budget))
        plot_marketing_optimizer(marketing_df, optimized_budget)

with tab4:
    st.header("V. AI-Driven Innovation Hub")
    st.markdown("_This section is at the forefront of integrating AI to enhance processes and accelerate business outcomes, moving from manual analysis to automated, intelligent systems._")
    st.subheader("A. AI Opportunity Recommendation Engine")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To proactively identify and score the best opportunities for AI-driven process automation across the organization, creating a data-driven roadmap for innovation.
        - **Method:** A scoring model evaluates potential AI projects based on two key axes: **Impact** (e.g., potential for cost savings, revenue generation, or risk reduction) and **Feasibility** (e.g., data quality, process standardization, API availability). The results are plotted on a 2x2 matrix to guide prioritization.
        - **Interpretation:** This engine automatically surfaces the highest-value opportunities. Items in the 'Quick Wins' quadrant are prime candidates for immediate resourcing. 'Strategic Bets' represent larger, more complex transformations that require executive sponsorship and a multi-quarter roadmap. This tool allows the leader to be at the forefront of integrating AI by presenting a clear, defensible plan for where to invest.
        """)
    st.plotly_chart(plot_ai_opportunity_matrix(ai_opp_df, segment_filter), use_container_width=True)

# ============================ SIDEBAR ============================
st.sidebar.markdown("### Key Responsibilities")
st.sidebar.markdown("""
- **Lead Weekly Sr. Leadership Meetings**
- **Drive Efficiency & Defect Elimination**
- **Manage Annual Planning Cycle**
- **Innovate with Tools & AI**
- **Tell Stories with Data**
""")
