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
min_value, max_value = None, None
x, y = None, None

@app.before_first_request
def initialize_app():
    global manager, min_value, max_value, x, y

    print("\nInitializing app...\n")
    manager = flask_run()
    print("\nManager initialized\n")

    print("Loading data...")
    x, y, min_value, max_value = load_data("loans", preprocessing="Normalize")
    print("Data loaded\n")


@app.route("/")
def index():
    x_explain = x[0:10]

    explain_preds = manager.predict(x_explain)
    instances, explanations = manager.explain(x_explain, explain_preds)

    explanations_dict = {"explanations": []}

    # Populate the dictionary
    for explanation in explanations:
        explanations_dict["explanations"].append(explanation)

    # Convert the dictionary to JSON
    json_explanations = json.dumps(explanations_dict, indent=4)

    return json.loads(json_explanations)


@app.route("/facet/explanation", methods=["POST"])
def facet_explanation():
    try:
        # example instance: 
        # {
        #    "ApplicantIncome": 4583,
        #    "CoapplicantIncome": 1508,
        #    "LoanAmount": 12800,
        #    "LoanAmountTerm": 360
        # }
        # label: N
        data = request.json

        # Extract and transform the input data into a numpy array
        applicant_income = data.get("ApplicantIncome", 0)
        coapplicant_income = data.get("CoapplicantIncome", 0)
        loan_amount = data.get("LoanAmount", 0)
        loan_amount_term = data.get("LoanAmountTerm", 0)

        input_data = np.array(
            [
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_amount_term
            ]
        )

        # Normalize the input data and reshape to 2d array
        input_data = (input_data - min_value) / (max_value - min_value)
        input_data = input_data.reshape(1, -1)

        # Perform explanations using manager.explain
        explain_pred = manager.predict(input_data)
        instance, explanation = manager.explain(input_data, explain_pred)

        return jsonify(explanation[0])
    
    except Exception as e:
        return jsonify({"error": str(e)})
    

if __name__ == "__main__":
    app.run(port=3000, debug=True)
