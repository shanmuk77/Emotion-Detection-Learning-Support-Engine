import streamlit as st
from config import EMOTIONS, MIXED_THRESHOLD
from models.bilstm_service import BiLSTMService
from models.bert_service import BERTService
from utils.emotion import primary_emotion, mixed_emotions
from utils.validation import validate_challenge
from services.gemini_service import generate_ai_response, support_strategy
from services.csv_store import append_interaction, load_history, nearest_saved_examples
from dashboard import render_dashboard

# Set page config
st.set_page_config(page_title="Emotion-Aware Learning Assistant", layout="wide")

# Caching model resource loading
@st.cache_resource
def load_bilstm():
    return BiLSTMService().load()

@st.cache_resource
def load_bert():
    return BERTService().load()

def score_panel(title, scores):
    st.subheader(title)
    primary = primary_emotion(scores)
    mixed = mixed_emotions(scores, MIXED_THRESHOLD)
    st.markdown("### " + " + ".join(mixed))
    st.success(f"Primary: {primary} ({scores[primary]:.1%})")
    
    # Sort and display all emotion scores
    for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        st.write(f"{emotion}: {score:.1%}")
        st.progress(min(max(float(score), 0.0), 1.0))
        
    return primary, mixed

# Title and subtitle
st.title("🤖 Emotion-Aware Learning Assistant")
st.caption("Get personalized help based on your field and emotional state")

# Tabs
main_tab, analytics_tab = st.tabs(["Learning Assistant", "Analytics"])

with main_tab:
    left, right = st.columns([2, 1])
    
    with left:
        st.header("Tell us about your learning challenge")
        field = st.selectbox(
            "What field are you studying?",
            ["Computer Science", "Mathematics", "Physics", "Engineering", "Other"]
        )
        problem = st.text_area(
            "Describe your problem or challenge",
            placeholder="Example: I'm confused about recursion"
        )
        st.caption("Quick examples: confused about recursion • debugging is frustrating • curious about ML")
        
    with right:
        st.header("Settings")
        use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
        save_csv = st.checkbox("Save to CSV for learning", value=True)
        show_details = st.checkbox("Show analysis details", value=True)
        use_saved = st.checkbox("Use CSV-based prediction", value=True)
        
        # Load history count dynamically
        try:
            history = load_history()
            history_count = len(history)
        except Exception:
            history_count = 0
        st.info(f"Using {history_count} saved examples for prediction")
        
    # Main action button
    if st.button("🔍 Get AI Learning Help", use_container_width=True, type="primary"):
        is_valid, err_msg = validate_challenge(problem)
        if not is_valid:
            st.warning(err_msg)
        else:
            try:
                # Load models
                bilstm = load_bilstm()
                bert = load_bert()
                
                # Predict
                bilstm_clean, bilstm_scores = bilstm.predict(problem)
                bert_clean, bert_scores = bert.predict(problem)
                
                st.header("📊 Model Predictions Comparison")
                c1, c2 = st.columns(2)
                
                with c1:
                    b_primary, b_mixed = score_panel("BiLSTM Student Adaptive", bilstm_scores)
                with c2:
                    t_primary, t_mixed = score_panel("BERT Transformer", bert_scores)
                
                confidence = bilstm_scores[b_primary]
                
                st.header("🤖 AI Learning Assistant Response")
                st.info(f"AI Response based on BiLSTM prediction: {b_primary}")
                
                # Generate AI Response
                if use_ai:
                    with st.spinner("Generating learning guidance..."):
                        response = generate_ai_response(field, problem, b_primary, confidence)
                else:
                    response = "AI response is disabled. Use the support strategy below."
                st.write(response)
                
                # Additional Support
                st.header("💡 Additional Support")
                strategy = support_strategy(b_primary)
                st.info("Strategy: " + strategy)
                
                # Show details if selected
                if show_details:
                    with st.expander("🔍 Analysis Details", expanded=True):
                        st.write("**Original Problem:**", problem)
                        st.write("**BiLSTM Processed:**", bilstm_clean)
                        st.write("**BiLSTM Confidence:**", round(confidence, 4))
                        st.write("**BERT Processed:**", bert_clean)
                        st.write(f"**AI Model Status:** BiLSTM: OK | BERT: OK | Gemini: {'Enabled' if use_ai else 'Disabled'}")
                        
                # Similarity matches
                if use_saved:
                    similar = nearest_saved_examples(problem, n=3)
                    if not similar.empty:
                        st.subheader("Similar Saved Learning Cases")
                        # Format similarity score as percentage
                        similar_display = similar.copy()
                        similar_display["similarity"] = similar_display["similarity"].map(lambda x: f"{x:.1%}")
                        st.dataframe(
                            similar_display[["field", "problem", "primary_emotion", "similarity"]],
                            use_container_width=True
                        )
                
                # Save to CSV
                if save_csv:
                    append_interaction(
                        field, problem, "BiLSTM",
                        " + ".join(b_mixed), b_primary, bilstm_scores[b_primary]
                    )
                    append_interaction(
                        field, problem, "BERT",
                        " + ".join(t_mixed), t_primary, bert_scores[t_primary]
                    )
                    st.success("Interaction saved to CSV successfully.")
                    
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Please train both models first using the commands:\n"
                        "1. `python training/train_bilstm.py`\n"
                        "2. `python training/train_bert.py`\n"
                        "Make sure you place training data in `data/emotions.csv` first.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

with analytics_tab:
    render_dashboard()
