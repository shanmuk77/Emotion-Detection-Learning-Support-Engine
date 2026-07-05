def validate_challenge(text: str):
    """Validates the student learning challenge text input."""
    if not text:
        return False, "Please enter a learning challenge."
        
    cleaned = text.strip()
    if not cleaned:
        return False, "Please enter a learning challenge."
        
    if len(cleaned) < 5:
        return False, "Please enter a longer challenge (at least 5 characters) so we can analyze your emotions accurately."
        
    if len(cleaned) > 1000:
        return False, "Your input is too long. Please shorten your challenge to under 1000 characters."
        
    return True, ""
