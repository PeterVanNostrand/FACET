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
test_applications = None
min_values, max_values = None, None


def init_app():
    global manager, test_applications, min_values, max_values

    print("\nApp initializing...\n")

    manager, test_applications, min_values, max_values = flask_run()

    for test_application in test_applications:
        for i in range(len(test_application)):
            min_val = min_values[i]
            max_val = max_values[i]
            test_application[i] = min_val + test_application[i] * (max_val - min_val)

    print("\nApp initialized\n")


init_app()


@app.route("/facet/applications", methods=["GET"])
def get_test_applications():
    num_arrays, array_length = test_applications.shape

    json_data = []

    # Iterate over the arrays and build the dictionary
    for i in range(num_arrays):
        values = [round(val, 0) for val in test_applications[i, :]]
        json_data.append(
            {
                "x0": values[0],
                "x1": values[1],
                "x2": values[2],
                "x3": values[3],
            }
        )

    return jsonify(json_data)


@app.route("/facet/explanation", methods=["POST"])
def facet_explanation():
    try:
        data = request.json

        # Extract and transform the input data into a numpy array
        applicant_income = data.get("x0", 0)
        coapplicant_income = data.get("x1", 0)
        loan_amount = data.get("x2", 0)
        loan_amount_term = data.get("x3", 0)
        num_explanations = data.get("num_explanations", 1)
        constraints = {
            "x0": [1000, 1600],
            "x1": [0, 10],
            "x2": [6000, 10000],
            "x3": [300, 500],
        }

        constraints = np.array([
            [1000, 1600],
            [0, 10],
            [6000, 10000],
            [300, 500]
        ])

        reshaped_mins = np.repeat(min_values, 2, axis=0)
        reshaped_mins = reshaped_mins.reshape(4, 2)
        reshaped_maxs = np.repeat(max_values, 2, axis=0)
        reshaped_maxs = reshaped_maxs.reshape(4, 2)

        input_data = np.array(
            [applicant_income, coapplicant_income, loan_amount, loan_amount_term]
        )

        # Normalize the input data and reshape to 2d array
        input_data = (input_data - min_values) / (max_values - min_values)
        input_data = input_data.reshape(1, -1)

        # Normalize the constraints
        normalized_constraints = (constraints - reshaped_mins) / (reshaped_maxs - reshaped_mins)

        # Perform explanations using manager.explain
        explain_pred = manager.predict(input_data)
        instance, explanations = manager.explain(
            input_data, explain_pred, num_explanations, normalized_constraints
        )

        new_explanations = []

        for explanation in explanations:
            new_values = {}

            for i, (feature, values) in enumerate(explanation.items()):
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

                new_values["x{:d}".format(i)] = [round(new_low, 1), round(new_high, 1)]

            new_explanations.append(new_values)

        # Update the original explanations with the modified values
        for i in range(len(explanations)):
            explanations[i].update(new_explanations[i])

        return jsonify(explanations)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(port=3001, debug=True)
