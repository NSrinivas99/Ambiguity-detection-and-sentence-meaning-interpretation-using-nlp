from flask import Flask, render_template, request, jsonify
from sentence_model import interpret_sentence   # ← correct import

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    result = interpret_sentence(text)   # ← correct function call

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
