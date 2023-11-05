from flask import Flask, request, jsonify
import numpy as np
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
facet_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(facet_dir)
from main import flask_run
from dataset import load_data

app = Flask(__name__)

manager = None


@app.before_first_request
def initialize_app():
    global manager
    print("\nInitializing app...\n")

    manager = flask_run()
    print("\nManager initialized\n")

def serialize_np_array(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Type not serializable")

@app.route("/")
def index():
    x, y = load_data("loans", preprocessing="Normalize")
    x_explain = x[0:10]

    explain_preds = manager.predict(x_explain)
    explanations = manager.explain(x_explain, explain_preds)

    explanations_dict = {
        "explanations": []
    }

    # Populate the dictionary
    for explanation in explanations:
        explanations_dict["explanations"].append(explanation)

    # Convert the dictionary to JSON
    json_explanations = json.dumps(explanations_dict, indent=4)

    return json.loads(json_explanations)


@app.route("/facet/explanation", methods=["POST"])
def facet_explanation():
    data = request.get_json()
    result = process_facet_data(data)

    return jsonify(result)


def process_facet_data(data):
    return data


if __name__ == "__main__":
    app.run(port=3000)
