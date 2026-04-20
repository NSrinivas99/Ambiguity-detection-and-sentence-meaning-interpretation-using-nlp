from flask import Flask, render_template, request, jsonify, session
from sentence_model import interpret_sentence

app = Flask(__name__)
app.secret_key = "ambigo-secret-key-2024"

@app.route("/")
def home():
    # Initialize accuracy tracking in session
    if "total" not in session:
        session["total"] = 0
        session["correct"] = 0
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    result = interpret_sentence(text)

    # Increment total count on each analysis
    session["total"] = session.get("total", 0) + 1

    total = session["total"]
    correct = session.get("correct", 0)
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0

    result["total"] = total
    result["correct"] = correct
    result["accuracy"] = accuracy

    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    is_correct = data.get("correct", False)

    if is_correct:
        session["correct"] = session.get("correct", 0) + 1

    total = session.get("total", 0)
    correct = session.get("correct", 0)
    accuracy = round((correct / total) * 100, 1) if total > 0 else 0

    return jsonify({
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    })

@app.route("/reset-accuracy", methods=["POST"])
def reset_accuracy():
    session["total"] = 0
    session["correct"] = 0
    return jsonify({"message": "Accuracy reset", "total": 0, "correct": 0, "accuracy": 0})

if __name__ == "__main__":
    app.run(debug=True)