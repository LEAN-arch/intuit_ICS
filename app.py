# ======================================================================================
# INTUIT DEMAND PLANNING COMMAND CENTER: EFFICIENCY & ANNUAL PLANNING
#
# A single-file Streamlit application for the Sr. Manager, Demand Planning.
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
#   - Lean Six Sigma & Root Cause Analysis
#   - Statistical Process Control (SPC) for Service Operations
#   - Machine Learning for Time-Series Forecasting & Driver Analysis
#   - Strategic Program Management & Risk Assessment
#
# To Run:
# 1. Save this code as 'intuit_demand_planning_final.py'
# 2. Create 'requirements.txt' with specified libraries.
# 3. Install dependencies: pip install -r requirements.txt
# 4. Run from your terminal: streamlit run intuit_demand_planning_final.py
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
st.set_page_config(
    page_title="Intuit Demand Planning Command Center",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)
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


# ======================================================================================
# SECTION 2: SME-DRIVEN DATA SIMULATION FOR INTUIT'S ECOSYSTEM
# ======================================================================================
@st.cache_data(ttl=600)
def generate_master_data():
    np.random.seed(42)
    # 1. Annual Operating Plan (AOP) Data
    aop_data = {
        'Metric': ['Prior Year Actuals', 'Organic Growth', 'Marketing & Sales Lift', 'New Product Launch', 'Platform Synergy Lift', 'Efficiency Savings', 'Final AOP Target'],
        'TurboTax_Users_M': [80, 5, 10, 3, 2, -1.5, 98.5],
        'QuickBooks_Subs_M': [8, 1.2, 2, 0.5, 0.8, -0.2, 12.3]
    }
    aop_df = pd.DataFrame(aop_data)

    # 2. Defect & Efficiency Data
    defects = ['Login Failure (High Freq)', 'Payment Error', 'QBO-Bank Sync Latency', 'Tax Form Import Bug', 'Mobile App Crash', 'Onboarding Drop-off', 'Live Expert Wait Time']
    defect_df = pd.DataFrame({
        'Defect_Category': defects,
        'Weekly_Contact_Volume': np.random.randint(500, 10000, 7),
        'Cost_Per_Contact_USD': np.random.uniform(15, 50, 7),
        'Owning_Group': ['Identity', 'Commerce', 'Connectivity', 'Consumer Group', 'Mobile Platform', 'Web Platform', 'Live Services']
    })
    defect_df['Weekly_Cost_Impact_USD'] = defect_df['Weekly_Contact_Volume'] * defect_df['Cost_Per_Contact_USD']

    # 3. Forecast Model Data
    forecast_dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=104, freq='W-SUN'))
    forecast_df = pd.DataFrame({'Week': forecast_dates})
    forecast_df['Marketing_Spend_M_USD'] = np.random.uniform(5, 20, 104) * (1 + np.sin(np.arange(104) * (2 * np.pi / 52)) * 0.5)
    forecast_df['Macro_Economic_Index'] = np.linspace(100, 95, 104) + np.random.normal(0, 1, 104)
    forecast_df['Competitor_SOV_Pct'] = np.random.uniform(20, 35, 104)
    forecast_df['Is_Tax_Season'] = ((forecast_df['Week'].dt.month.isin([1,2,3,4]))).astype(int)
    forecast_df['New_Signups_K'] = (forecast_df['Marketing_Spend_M_USD']*10 - forecast_df['Competitor_SOV_Pct']*2 + forecast_df['Macro_Economic_Index']*0.5 + forecast_df['Is_Tax_Season']*50 + np.random.normal(0, 10, 104) + 50).clip(20)

    # 4. Clickstream Funnel Data for a key process
    funnel_data = {
        'Stage': ['1. Visited Landing Page', '2. Started Signup', '3. Entered Credentials', '4. Verified Email', '5. Completed Onboarding'],
        'Users': [5000000, 3500000, 3000000, 2200000, 2000000]
    }
    funnel_df = pd.DataFrame(funnel_data)

    return aop_df, defect_df, forecast_df, funnel_df

# ======================================================================================
# SECTION 3: ADVANCED ANALYTICAL & ML MODELS
# ======================================================================================
@st.cache_resource
def get_forecast_model(df):
    features = ['Marketing_Spend_M_USD', 'Macro_Economic_Index', 'Competitor_SOV_Pct', 'Is_Tax_Season']
    target = 'New_Signups_K'
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=3, max_depth=10)
    model.fit(X_train, y_train)
    importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    return model, X_test, y_test, importance

def plot_aop_waterfall(df, segment, unit):
    fig = go.Figure(go.Waterfall(
        name=segment, orientation="v",
        measure=["absolute"] + ["relative"] * 5 + ["total"],
        x=df['Metric'],
        text=[f"{v:,.1f}{unit}" for v in df[segment]],
        y=df[segment],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"#D92D20"}},
        increasing={"marker":{"color":"#00855A"}},
        totals={"marker":{"color":"#0077C5"}}
    ))
    fig.update_layout(title=f"<b>Annual Operating Plan (AOP) Build: {segment}</b>", yaxis_title=f"Users / Subscriptions ({unit})")
    return fig

def plot_defect_priority_matrix(df):
    avg_volume = df['Weekly_Contact_Volume'].mean()
    avg_cost = df['Weekly_Cost_Impact_USD'].mean()
    fig = px.scatter(
        df, x='Weekly_Contact_Volume', y='Weekly_Cost_Impact_USD',
        size='Weekly_Cost_Impact_USD', color='Owning_Group',
        text='Defect_Category', hover_name='Defect_Category',
        title='<b>Defect Prioritization Matrix: Frequency vs. Financial Impact</b>',
        labels={'Weekly_Contact_Volume': 'Frequency (Weekly Contacts)', 'Weekly_Cost_Impact_USD': 'Financial Impact (Weekly Cost USD)'}
    )
    fig.update_traces(textposition='top center')
    fig.add_vline(x=avg_volume, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=avg_cost, line_width=1, line_dash="dash", line_color="grey")
    fig.add_annotation(x=avg_volume*1.5, y=avg_cost*1.5, text="<b>🔥 TOP PRIORITY 🔥</b>", showarrow=False, font=dict(color="#D92D20", size=14))
    fig.add_annotation(x=avg_volume*0.5, y=avg_cost*1.5, text="<b>High Cost / Low Freq.</b>", showarrow=False, font=dict(color="grey"))
    fig.add_annotation(x=avg_volume*1.5, y=avg_cost*0.5, text="<b>High Freq. / Low Cost</b>", showarrow=False, font=dict(color="grey"))
    return fig

def plot_clickstream_funnel(df):
    df['Dropoff_From_Previous'] = df['Users'].diff().abs().fillna(0)
    df['Dropoff_Rate'] = (df['Dropoff_From_Previous'] / df['Users'].shift(1) * 100).fillna(0)
    fig = go.Figure(go.Funnel(
        y=df['Stage'],
        x=df['Users'],
        texttemplate="%{value:,d} <br>Dropoff: %{customdata[0]:.1f}%",
        textinfo="value",
        customdata=df[['Dropoff_Rate']]
    ))
    fig.update_layout(title="<b>Clickstream Funnel Analysis: New User Onboarding</b>")
    return fig

# ======================================================================================
# SECTION 4: MAIN APPLICATION LAYOUT & SCIENTIFIC NARRATIVE
# ======================================================================================
st.title("💡 Intuit Demand Planning Command Center")
st.markdown("##### A strategic dashboard for the Sr. Manager of Efficiency & Annual Planning, focused on driving efficiency, eliminating defects, and integrating AI.")
aop_df, defect_df, forecast_df, funnel_df = generate_master_data()
forecast_model, X_test, y_test, importance_df = get_forecast_model(forecast_df)

st.markdown("### I. Executive Command Center (Weekly Leadership View)")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
total_efficiency_savings = aop_df[aop_df['Metric'] == 'Efficiency Savings'][['TurboTax_Users_M', 'QuickBooks_Subs_M']].sum().sum() * -1
kpi_col1.metric("AOP Efficiency Target", f"${total_efficiency_savings*100:.0f}M+", help="Total annualized savings baked into the Annual Operating Plan from defect elimination and process improvements. Assumes $100 LTV per user/sub.")
top_defect_cost = defect_df.sort_values('Weekly_Cost_Impact_USD', ascending=False)['Weekly_Cost_Impact_USD'].iloc[0] * 52
kpi_col2.metric("Top Defect Annualized Cost", f"${top_defect_cost/1_000_000:.2f}M", help="Annual cost of the #1 most expensive defect ('Login Failure'), representing the largest single efficiency opportunity.")
mape = 100 * np.mean(np.abs(forecast_model.predict(X_test) - y_test) / y_test)
kpi_col3.metric("Strategic Forecast Accuracy (MAPE)", f"{mape:.1f}%", "-0.5% vs last quarter", "inverse", help="Mean Absolute Percentage Error on the holdout test set for the new user forecast model. Lower is better.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["**II. ANNUAL OPERATING PLAN (AOP)**", "**III. EFFICIENCY & DEFECT ELIMINATION**", "**IV. STRATEGIC FORECASTING & SCENARIO PLANNING**", "**V. AI-DRIVEN INNOVATION**"])

with tab1:
    st.header("II. Annual Operating Plan (AOP) Storytelling")
    st.markdown("_This view provides the end-to-end narrative of the annual plan, ideal for presenting to senior leadership. It clearly articulates the build from prior year actuals to the final target, highlighting key assumptions and the value of strategic initiatives._")
    st.subheader("A. AOP Waterfall Decomposition")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To deconstruct the Annual Operating Plan into its core components, telling a clear, logical story of how the final target was derived. This is the primary tool for driving cross-functional alignment during the bottoms-up planning cycle.
        - **Method:** A waterfall chart is used to visualize the sequential build-up (or build-down) of the plan. It starts with a baseline (Prior Year Actuals) and adds or subtracts the forecasted impact of key drivers. 'Efficiency Savings' is shown as a negative value, representing a cost-avoidance or demand-reduction benefit.
        - **Interpretation:** This visualization transforms a complex spreadsheet into an intuitive narrative. It allows leaders to instantly grasp the key levers of the plan (e.g., "Marketing is our biggest growth driver") and the magnitude of each assumption. It also explicitly highlights the value of the Efficiency function by showing its direct contribution to the overall plan.
        """)
    seg_choice = st.radio("Select Business Segment:", ('TurboTax', 'QuickBooks'), horizontal=True)
    if seg_choice == 'TurboTax':
        st.plotly_chart(plot_aop_waterfall(aop_df, 'TurboTax_Users_M', 'M'), use_container_width=True)
    else:
        st.plotly_chart(plot_aop_waterfall(aop_df, 'QuickBooks_Subs_M', 'M'), use_container_width=True)

with tab2:
    st.header("III. Efficiency & Defect Elimination Engine")
    st.markdown("_This section is the analytical core for identifying, prioritizing, and driving accountability for fixing customer-facing defects that create operational drag._")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("A. Defect Prioritization Matrix")
        with st.expander("View Methodological Summary", expanded=True):
            st.markdown("""
            - **Purpose:** To move beyond simple volume-based prioritization and identify the most strategically important defects to eliminate, based on both customer friction (frequency) and business cost (financial impact).
            - **Method:** A 2x2 matrix plotting Weekly Contact Volume vs. Weekly Cost Impact. The size of each bubble represents the total cost, and the color represents the owning programmatic team.
            - **Interpretation:** The top-right quadrant, **'Top Priority'**, contains defects that are both frequent and expensive. These are the highest-value targets for elimination. This matrix is a powerful tool for "driving accountability with Program leaders" by clearly showing which teams own the most critical defects and quantifying the 'prize' for fixing them.
            """)
        st.plotly_chart(plot_defect_priority_matrix(defect_df), use_container_width=True)
    with col2:
        st.subheader("B. Clickstream Funnel Analysis")
        with st.expander("View Methodological Summary", expanded=True):
            st.markdown("""
            - **Purpose:** To instrument and analyze a key customer journey to pinpoint sources of friction and drop-off, a common source of "invisible" defects that don't always result in a support contact.
            - **Method:** A funnel chart visualizes the user drop-off at each stage of a multi-step process (e.g., onboarding). It calculates and displays the conversion rate from one step to the next.
            - **Interpretation:** The largest percentage drop-off identifies the biggest point of friction in the process flow. In this example, the drop from 'Entered Credentials' to 'Verified Email' (26.7% drop-off) is the most significant leak in the funnel. This provides a specific, data-driven mandate for the Identity and Onboarding teams to investigate and simplify this step.
            """)
        st.plotly_chart(plot_clickstream_funnel(funnel_df), use_container_width=True)

with tab3:
    st.header("IV. Strategic Forecasting & Interactive Scenario Planning")
    st.markdown("_This section provides thought partnership to business leaders by moving beyond a single forecast to an interactive planning tool, enabling data-driven decisions on resource allocation._")
    st.subheader("A. Interactive Scenario Planner")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To create an interactive tool that allows business leaders to simulate the impact of strategic decisions on key outcomes (New Signups). This transforms the planning function from a reporting entity to a strategic advisor.
        - **Method:** The underlying Random Forest forecast model is used to generate predictions in real-time based on user-adjusted inputs from the sliders below. The chart displays the resulting forecast against a baseline, immediately quantifying the impact of the simulated changes.
        - **Interpretation:** This tool facilitates data-driven conversations about trade-offs. A leader can ask, "What's the ROI on an additional $2M in marketing spend?" and see the projected lift instantly. It allows for testing hypotheses and aligning on a final set of assumptions for the AOP based on quantitative analysis.
        """)
    col1, col2, col3 = st.columns(3)
    marketing_adj = col1.slider("Marketing Spend Adjustment (%)", -50, 50, 0)
    macro_adj = col2.slider("Macro-Economic Index Adjustment (pts)", -10, 10, 0)
    competitor_adj = col3.slider("Competitor SOV Adjustment (pts)", -10, 10, 0)
    baseline_pred = forecast_model.predict(X_test)
    adjusted_X = X_test.copy()
    adjusted_X['Marketing_Spend_M_USD'] *= (1 + marketing_adj/100)
    adjusted_X['Macro_Economic_Index'] += macro_adj
    adjusted_X['Competitor_SOV_Pct'] += competitor_adj
    adjusted_pred = forecast_model.predict(adjusted_X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=baseline_pred, name='Baseline Forecast', mode='lines', line=dict(color='grey', dash='dash')))
    fig.add_trace(go.Scatter(x=y_test.index, y=adjusted_pred, name='Scenario Forecast', mode='lines', line=dict(color='blue', width=3)))
    fig.update_layout(title='<b>Forecast Scenario vs. Baseline</b>', yaxis_title='New Signups (Thousands)', xaxis_title='Week')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("V. AI-Driven Innovation & Anomaly Detection")
    st.markdown("_This section showcases the integration of AI to move from reactive analysis to proactive intelligence, automatically surfacing risks and opportunities._")
    st.subheader("A. AI-Powered Anomaly Detection in New Signups")
    with st.expander("View Methodological Summary", expanded=True):
        st.markdown("""
        - **Purpose:** To automatically monitor the most critical top-of-funnel metric (New Signups) for statistically significant deviations from the expected trend, enabling faster response to unforeseen market shifts or system defects.
        - **Method:** A rolling statistical process control (SPC) logic is applied to the time-series data. The model calculates a rolling average and standard deviation over a defined window (e.g., 12 weeks). An 'anomaly' is flagged if a data point falls outside of a specified threshold (e.g., ±3 standard deviations) from the rolling mean.
        - **Interpretation:** The red dots automatically flag weeks where signups were statistically unusual. This serves as an early warning system. An unexpected negative anomaly could indicate a critical bug in the signup flow, a competitor's surprise campaign, or a website outage. An unexpected positive anomaly might highlight a viral social media mention or a highly effective new ad creative that should be doubled-down on. This automates insight generation and accelerates business response.
        """)
    df = forecast_df.copy()
    df = df.set_index('Week')
    rolling_mean = df['New_Signups_K'].rolling(window=12).mean()
    rolling_std = df['New_Signups_K'].rolling(window=12).std()
    df['upper_band'] = rolling_mean + (rolling_std * 3)
    df['lower_band'] = rolling_mean - (rolling_std * 3)
    anomalies = df[(df['New_Signups_K'] > df['upper_band']) | (df['New_Signups_K'] < df['lower_band'])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['New_Signups_K'], name='Actual New Signups', mode='lines', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], name='Anomaly Threshold', mode='lines', line=dict(color='rgba(0,119,197,0.3)')))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], name='Anomaly Threshold', mode='lines', line=dict(color='rgba(0,119,197,0.3)'), fill='tonexty'))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['New_Signups_K'], mode='markers', name='Anomaly Detected', marker=dict(color='red', size=10, symbol='x')))
    fig.update_layout(title='<b>Automated Anomaly Detection on New Signups</b>', yaxis_title='New Signups (Thousands)', xaxis_title='Week')
    st.plotly_chart(fig, use_container_width=True)

# ============================ SIDEBAR ============================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Intuit_logo.svg/1280px-Intuit_logo.svg.png", use_container_width=True)
st.sidebar.markdown("### Role Focus")
st.sidebar.info("This dashboard is for the **Sr. Manager, Demand Planning (Efficiency & Annual Planning)**, responsible for end-to-end efficiency, defect elimination, and annualized planning cycles at Intuit.")
st.sidebar.markdown("### Key Responsibilities")
st.sidebar.markdown("""
- **Lead Weekly Sr. Leadership Meetings:** Present demand/supply plans, gaps, and risks.
- **Drive Efficiency & Defect Elimination:** Use data to find and eliminate sources of waste.
- **Manage Annual Planning Cycle:** Facilitate bottoms-up planning across the organization.
- **Innovate with Tools & AI:** Build and deploy new methodologies to improve forecast accuracy and streamline workflows.
- **Tell Stories with Data:** Influence and drive momentum with stakeholders and leaders.
""")
