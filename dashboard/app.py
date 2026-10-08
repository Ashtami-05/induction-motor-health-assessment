import pandas as pd
import streamlit as st
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="Induction Motor Health Assessment",
    page_icon="⚙️",
    layout="wide"
)


# Title
st.title("⚙️ Induction Motor Health Assessment Dashboard")
st.write(
    "Health assessment and maintenance decision support "
    "based on vibration signal classification."
)


# Load results
file_path = "health_assessment/health_assessment_results.csv"
df = pd.read_csv(file_path)


# Summary values
total_samples = len(df)
high_risk = (df["risk_level"] == "High").sum()
medium_risk = (df["risk_level"] == "Medium").sum()
low_risk = (df["risk_level"] == "Low").sum()
average_health = df["health_score"].mean()


# Display summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Samples", total_samples)

with col2:
    st.metric("Average Health Score", f"{average_health:.1f}")

with col3:
    st.metric("High Risk", high_risk)

with col4:
    st.metric("Low Risk", low_risk)


st.divider()


# Risk-level distribution
st.subheader("⚠️ Risk Level Distribution")

risk_counts = df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]

fig_risk = px.bar(
    risk_counts,
    x="Risk Level",
    y="Count",
    title="Number of Samples by Risk Level"
)

st.plotly_chart(fig_risk, width="stretch")


# Predicted fault distribution
st.subheader("🔍 Predicted Fault Distribution")

fault_counts = df["predicted_fault"].value_counts().reset_index()
fault_counts.columns = ["Fault Type", "Count"]

fig_fault = px.pie(
    fault_counts,
    names="Fault Type",
    values="Count",
    title="Predicted Bearing Conditions"
)

st.plotly_chart(fig_fault, width="stretch")


# Health score distribution
# Overall health score gauge
st.subheader("❤️ Overall Motor Health Score")

fig_gauge = px.pie(
    values=[average_health, 100 - average_health],
    names=["Health Score", "Remaining"],
    hole=0.7,
    title=f"Overall Health Score: {average_health:.1f}/100"
)

fig_gauge.update_traces(
    textinfo="none"
)

fig_gauge.update_layout(
    showlegend=False,
    annotations=[
        dict(
            text=f"{average_health:.1f}/100",
            x=0.5,
            y=0.5,
            font=dict(size=28),
            showarrow=False
        )
    ]
)

st.plotly_chart(fig_gauge, width="stretch")
# Fault condition filter
st.subheader("🔍 Filter Results by Fault Condition")

fault_options = ["All"] + sorted(df["predicted_fault"].unique().tolist())

selected_fault = st.selectbox(
    "Select fault condition:",
    fault_options
)

if selected_fault == "All":
    filtered_df = df
else:
    filtered_df = df[df["predicted_fault"] == selected_fault]

st.write(f"Showing {len(filtered_df)} samples")

st.dataframe(
    filtered_df,
    width="stretch"
)
# Detailed results
st.subheader("📋 Detailed Assessment Results")

st.dataframe(
    df,
    use_container_width=True
)