from flask import Flask, jsonify, request

from agent import ask

app = Flask(__name__)

print("✔ Flask app is loading...")


@app.route("/", methods=["POST"])
def answer_question():
    payload = request.get_json(silent=True)
    question = payload.get("question") if isinstance(payload, dict) else None

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": 'Expected a JSON body like {"question": "How many passengers survived?"}'}), 400

    try:
        result = ask(question.strip())
    except Exception as e:
        # e.g. the model provider failing or the agent hitting its recursion limit
        app.logger.exception("agent failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502

    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200
