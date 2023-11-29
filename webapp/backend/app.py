from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json, os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
facet_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(facet_dir)
from main import flask_run
from dataset import load_data
from config import VIZ_DATA_PATH
app = Flask(__name__)
CORS(app)

manager = None
test_applications = None
min_values, max_values = None, None
infinity = 100000000000000
port = 3001

#Master json file
json_path = "visualization\data\dataset_details.json"
master_json = None


def init_app():
    global manager, test_applications, min_values, max_values, reshaped_mins, reshaped_maxs

    print("\nApp initializing...\n")

    manager, test_applications, min_values, max_values = flask_run()

    reshaped_mins = np.repeat(min_values, 2, axis=0).reshape(4, 2)
    reshaped_maxs = np.repeat(max_values, 2, axis=0).reshape(4, 2)

    for test_application in test_applications:
        for i in range(len(test_application)):
            min_val = min_values[i]
            max_val = max_values[i]
            test_application[i] = min_val + test_application[i] * (max_val - min_val)

    print("\nApp initialized\n")

init_app()


@app.route("/visualization/data/<path:filename>")
def serve_file(filename):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_dir = os.path.join(root_dir, "..", "..", "visualization")
    return send_from_directory(os.path.join(visualization_dir, "data"), filename)


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


@app.route("/facet/explanations", methods=["POST"])
def facet_explanation():
    try:
        data = request.json

        # Extract and transform the input data from the request
        applicant_income = data.get("x0", 0)
        coapplicant_income = data.get("x1", 0)
        loan_amount = data.get("x2", 0)
        loan_amount_term = data.get("x3", 0)
        num_explanations = data.get("num_explanations", 1)
        constraints = data.get("constraints", None)

        constraints = np.array(constraints)

        input_data = np.array(
            [applicant_income, coapplicant_income, loan_amount, loan_amount_term]
        )

        # Normalize the input data and constraints
        input_data = (input_data - min_values) / (max_values - min_values)
        input_data = input_data.reshape(1, -1)
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

                new_values["x{:d}".format(i)] = [round(new_low, 0), round(new_high, 0)]

            new_explanations.append(new_values)

        # Update the original explanations with the modified values
        for i in range(len(explanations)):
            explanations[i].update(new_explanations[i])

        return jsonify(explanations)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(port=port, debug=True)
