from flask import Flask, request, jsonify, render_template
import pickle
import os
import re
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
VECTORIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "vectorizer.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/verify")
def verify():
    return render_template("verify.html")


@app.route("/about")
def about():
    return render_template("about.html")


def analyze_text_features(text):
    """Analyze text features to provide reasoning for predictions"""
    features = {}

    # Convert to lowercase for analysis
    text_lower = text.lower()

    # 1. Sensationalist words/phrases
    sensationalist_words = [
        "shocking",
        "amazing",
        "incredible",
        "unbelievable",
        "mind-blowing",
        "you won't believe",
        "this will shock you",
        "breaking",
        "urgent",
        "exclusive",
        "secret",
        "hidden",
        "conspiracy",
        "cover-up",
    ]
    sensationalist_count = sum(1 for word in sensationalist_words if word in text_lower)
    features["sensationalist_language"] = sensationalist_count

    # 2. Emotional language
    emotional_words = [
        "outrageous",
        "disgusting",
        "terrifying",
        "horrifying",
        "amazing",
        "fantastic",
        "incredible",
        "wonderful",
        "terrible",
        "awful",
    ]
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    features["emotional_language"] = emotional_count

    # 3. Exclamation marks
    exclamation_count = text.count("!")
    features["exclamation_marks"] = exclamation_count

    # 4. ALL CAPS words
    caps_words = len(re.findall(r"\b[A-Z]{2,}\b", text))
    features["caps_words"] = caps_words

    # 5. Question marks (indicating uncertainty)
    question_count = text.count("?")
    features["question_marks"] = question_count

    # 6. Text length
    features["text_length"] = len(text)

    # 7. Word count
    word_count = len(text.split())
    features["word_count"] = word_count

    # 8. Average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in text.split()) / word_count
        features["avg_word_length"] = avg_word_length
    else:
        features["avg_word_length"] = 0

    # 9. Number of sentences
    sentences = re.split(r"[.!?]+", text)
    features["sentence_count"] = len([s for s in sentences if s.strip()])

    # 10. Presence of numbers (factual content)
    numbers = re.findall(r"\d+", text)
    features["number_count"] = len(numbers)

    return features


def generate_reasons(features, prediction, confidence):
    """Generate human-readable reasons based on analyzed features"""
    reasons = []

    # High sensationalist language
    if features["sensationalist_language"] > 2:
        reasons.append(
            {
                "type": "warning",
                "text": f"Contains {features['sensationalist_language']} sensationalist words/phrases",
                "explanation": "Sensationalist language is often used in fake news to grab attention and evoke emotional responses.",
            }
        )

    # High emotional language
    if features["emotional_language"] > 3:
        reasons.append(
            {
                "type": "warning",
                "text": f"Uses {features['emotional_language']} emotionally charged words",
                "explanation": "Excessive emotional language can indicate bias and manipulation rather than objective reporting.",
            }
        )

    # Too many exclamation marks
    if features["exclamation_marks"] > 2:
        reasons.append(
            {
                "type": "warning",
                "text": f"Contains {features['exclamation_marks']} exclamation marks",
                "explanation": "Professional news articles typically use exclamation marks sparingly.",
            }
        )

    # ALL CAPS words
    if features["caps_words"] > 1:
        reasons.append(
            {
                "type": "warning",
                "text": f"Contains {features['caps_words']} words in ALL CAPS",
                "explanation": "Excessive use of capital letters is often associated with clickbait and fake news.",
            }
        )

    # Too many questions
    if features["question_marks"] > 3:
        reasons.append(
            {
                "type": "warning",
                "text": f"Contains {features['question_marks']} question marks",
                "explanation": "Excessive questioning can indicate uncertainty or sensationalism rather than factual reporting.",
            }
        )

    # Positive indicators
    if features["number_count"] > 2:
        reasons.append(
            {
                "type": "positive",
                "text": f"Contains {features['number_count']} specific numbers/facts",
                "explanation": "Factual content with specific numbers often indicates more reliable reporting.",
            }
        )

    if features["text_length"] > 500:
        reasons.append(
            {
                "type": "positive",
                "text": f"Article length: {features['text_length']} characters",
                "explanation": "Longer articles often provide more context and detailed information.",
            }
        )

    if features["sentence_count"] > 5:
        reasons.append(
            {
                "type": "positive",
                "text": f"Contains {features['sentence_count']} sentences",
                "explanation": "Multiple sentences provide more context and detailed information.",
            }
        )

    # Confidence-based reasoning
    if confidence > 0.8:
        reasons.append(
            {
                "type": "confidence",
                "text": f"High confidence: {confidence:.1%}",
                "explanation": "The model is very confident in this prediction based on the analyzed features.",
            }
        )
    elif confidence < 0.6:
        reasons.append(
            {
                "type": "uncertainty",
                "text": f"Low confidence: {confidence:.1%}",
                "explanation": "The model shows uncertainty, suggesting mixed signals in the content.",
            }
        )

    return reasons


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    text = data.get("text", "")

    # Combine title and text
    combined = title + " " + text

    # Transform text
    features = vectorizer.transform([combined])

    # Make prediction
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    # Analyze text features for reasoning
    text_features = analyze_text_features(combined)
    confidence = float(max(proba))
    reasons = generate_reasons(text_features, prediction, confidence)

    # Return enhanced result
    result = {
        "prediction": "True News" if prediction == 1 else "Fake News",
        "confidence": confidence,
        "true_probability": float(proba[1]),
        "fake_probability": float(proba[0]),
        "reasons": reasons,
        "features": text_features,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
