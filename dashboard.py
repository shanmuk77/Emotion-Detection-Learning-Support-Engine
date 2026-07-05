import streamlit as st
import plotly.express as px
from services.csv_store import load_history

def render_dashboard():
    st.header("Analytics Dashboard")
    df = load_history()
    
    if df.empty:
        st.info("No saved interactions yet. Submit a learning challenge first to populate dashboard metrics!")
        return

    tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])
    
    with tab1:
        st.subheader("Emotion Distribution & Trends")
        
        # 1. Pie Chart for Emotion Distribution
        # Use primary_emotion or emotion (which stores mixed emotions like 'Bored + Frustrated' or just primary)
        counts = df["primary_emotion"].value_counts().reset_index()
        counts.columns = ["emotion", "count"]
        fig = px.pie(
            counts, 
            names="emotion", 
            values="count", 
            title="Primary Emotion Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2. Line Chart for Emotional Journey
        journey = df.copy()
        journey["timestamp"] = journey["timestamp"].astype(str)
        # Sort by timestamp to ensure chronological order in the line chart
        journey = journey.sort_values("timestamp")
        fig2 = px.line(
            journey, 
            x="timestamp", 
            y="confidence",
            color="primary_emotion", 
            markers=True, 
            title="Emotional Journey (Confidence Over Time)",
            labels={"primary_emotion": "Emotion", "confidence": "Confidence Score", "timestamp": "Time"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Field & Model Deep Dive")
        
        # Emotions by Study Field & Model
        grouped = (
            df.groupby(["field", "model", "primary_emotion"])
            .size()
            .reset_index(name="count")
        )
        fig3 = px.bar(
            grouped, 
            x="field", 
            y="count", 
            color="primary_emotion",
            facet_col="model", 
            barmode="group",
            title="Emotions by Study Field & Prediction Model",
            labels={"field": "Study Field", "count": "Interaction Count", "primary_emotion": "Emotion"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Performance & Usage Summary")
        
        # Compute summary metrics
        total_interactions = len(df)
        
        mode_emotion = "None"
        if not df["primary_emotion"].mode().empty:
            mode_emotion = df["primary_emotion"].mode().iloc[0]
            
        avg_confidence = df["confidence"].mean()
        
        most_used_field = "None"
        if not df["field"].mode().empty:
            most_used_field = df["field"].mode().iloc[0]
            
        bilstm_count = len(df[df["model"] == "BiLSTM"])
        bert_count = len(df[df["model"] == "BERT"])
        
        # Display Metrics in columns
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Interactions", total_interactions)
        c2.metric("Most Common Emotion", mode_emotion)
        c3.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("Most Active Field", most_used_field)
        c5.metric("BiLSTM Predictions", bilstm_count)
        c6.metric("BERT Predictions", bert_count)
        
        st.subheader("Recent Records")
        st.dataframe(df.tail(20), use_container_width=True)
