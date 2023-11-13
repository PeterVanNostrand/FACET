from flask import Flask, request, jsonify
from flask_cors import CORS
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
CORS(app)

manager = None
test_applicants = None
min_values, max_values = None, None
x, y = None, None
instances = None


def init_app():
    global manager, test_applicants, min_values, max_values, x, y, initialized
    print("\nInitializing app...")

    print("\nInitializing manager...")
    manager, test_applicants, x, y, min_values, max_values = flask_run()
    print("\nManager initialized\n")

init_app()

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
        data = request.json

        # Extract and transform the input data into a numpy array
        applicant_income = data.get("x0", 0)
        coapplicant_income = data.get("x1", 0)
        loan_amount = data.get("x2", 0)
        loan_amount_term = data.get("x3", 0)

        input_data = np.array(
            [applicant_income, coapplicant_income, loan_amount, loan_amount_term]
        )

        # Normalize the input data and reshape to 2d array
        # denormalize: output_data * (max_value - min_value) + min_value
        input_data = (input_data - min_values) / (max_values - min_values)
        input_data = input_data.reshape(1, -1)

        # Perform explanations using manager.explain
        explain_pred = manager.predict(input_data)
        instance, explanations = manager.explain(input_data, explain_pred)

        explanation = explanations[0]
        denormalized_explanation = {}

        for i, (feature, values) in enumerate(explanation.items()):
            print(i)
            min_val = min_values[i]
            max_val = max_values[i]
            low = values[0]
            high = values[1]

            new_low = (
                min_val
                if low == -100000000000000
                else min_val + low * (max_val - min_val)
            )
            new_high = (
                max_val
                if high == 100000000000000
                else min_val + high * (max_val - min_val)
            )
            
            # TODO round to 1 decimal place?
            denormalized_explanation["x{:d}".format(i)] = [
                round(new_low, 1),
                round(new_high, 1),
            ]


        return jsonify(denormalized_explanation)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(port=3001, debug=True)
