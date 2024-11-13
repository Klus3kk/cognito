from models.nlp_model import analyze_readability

def test_analyze_readability():
    code = "def myfunction():\n    return 1 + 1"
    feedback = analyze_readability(code)
    assert "Consider improving readability" in feedback or "Code readability looks good." in feedback
