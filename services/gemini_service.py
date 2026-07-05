import google.generativeai as genai
from config import GEMINI_API_KEY

FALLBACKS = {
    "Bored": "Try a 10-minute interactive example, then solve one tiny challenge.",
    "Frustrated": "Reduce the task to one small step and test each step separately.",
    "Confused": "Start with a plain-language definition, then a worked example.",
    "Curious": "Explore a deeper example and connect it to a real-world application.",
    "Confident": "Attempt a harder variation and explain your reasoning."
}

def support_strategy(emotion):
    mapping = {
        "Bored": "Show interactive content",
        "Frustrated": "Break task into smaller steps",
        "Confused": "Explain step by step with an example",
        "Curious": "Offer advanced resources and exploration",
        "Confident": "Give a challenging extension problem",
    }
    return mapping.get(emotion, "Provide structured learning support")

def generate_ai_response(field, problem, emotion, confidence):
    if not GEMINI_API_KEY:
        return FALLBACKS.get(emotion, "Work through one small step at a time.")
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
You are an emotion-aware learning assistant.
Study field: {field}
Student challenge: {problem}
Detected emotion: {emotion}
Confidence: {confidence:.1%}

Give concise, supportive, field-aware, actionable learning guidance.
Do not diagnose mental health. Include 3 concrete next steps.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # If gemini-2.5-flash fails or is not supported, try falling back to gemini-1.5-flash
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            return FALLBACKS.get(emotion, "Work through one small step at a time.")
