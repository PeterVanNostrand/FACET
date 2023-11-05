from flask import Flask, request, jsonify

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
facet_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(facet_dir)
from main import flask_run
from dataset import load_data

app = Flask(__name__)

manager = None


def test_manager():
    x, y = load_data("vertebral", preprocessing="Normalize")
    print(x, y)

    test_data = {"property_name": "sample_property"}

    result = manager.explainer.explain(test_data)
    print("Manager test result:", result)


@app.before_first_request
def initialize_app():
    print("Initializing app...")
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    manager = flask_run()
    print("Manager initialized")

    test_manager()


@app.route("/")
def index():
    return "Welcome to the Facet web app!"


@app.route("/facet/explanation", methods=["POST"])
def facet_explanation():
    data = request.get_json()
    result = process_facet_data(data)

    return jsonify(result)


def process_facet_data(data):
    # JSON object received in the request
    # data['property_name']

    return data


if __name__ == "__main__":
    app.run(debug=True)
