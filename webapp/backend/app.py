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
from config import VIZ_DATA_PATH
app = Flask(__name__)
CORS(app)

manager = None
test_applications = None
min_values, max_values = None, None
infinity = 100000000000000
port = 3001

#Master json file
json_path = f"{VIZ_DATA_PATH}\\dataset_details.json"
master_json = None


def init_app():
    global manager, test_applications, min_values, max_values, master_json

    print("\nApp initializing...\n")

    manager, test_applications, min_values, max_values = flask_run()

    for test_application in test_applications:
        for i in range(len(test_application)):
            min_val = min_values[i]
            max_val = max_values[i]
            test_application[i] = min_val + test_application[i] * (max_val - min_val)
    
    print("\nExtracting master json file details...\n")
    try:
        with open(json_path) as master:
            master_json = json.load(master)
            print("master_json file loaded")
    except Exception as e:
        print(f"ERROR: Unable to load master json file. Details:\n{e}")
        exit(1)

    print("\nApp initialized\n")


init_app()


@app.route("/facet/applications", methods=["GET"])
def get_test_applications():
    num_arrays, array_length = test_applications.shape

    json_data = []

    # Iterate over the arrays and build the dictionary
    for i in range(num_arrays):
        values = [round(val, 0) for val in test_applications[i, :]]
        val_dict = {} #dictionary that holds x0 - xn values
        for entry in range(array_length):
            val_dict[f"x{entry}"] = values[entry]
        
        json_data.append(val_dict)

    return jsonify(json_data)


@app.route("/facet/explanation", methods=["POST"])
def facet_explanation():
    try:
        data = request.json

        # Extract and transform the input data into a numpy array
        input_data = []
        for feature in master_json["feature_names"]:
            input_data.append(data.get(feature, 0))
        input_data = np.array(input_data)

        print("input_data", input_data) #debug

        # Normalize the input data and reshape to 2d array
        input_data = (input_data - min_values) / (max_values - min_values)
        input_data = input_data.reshape(1, -1)

        # Perform explanations using manager.explain
        explain_pred = manager.predict(input_data)
        instance, explanations = manager.explain(input_data, explain_pred)

        explanation = explanations[0]
        denormalized_explanation = {}

        for i, (feature, values) in enumerate(explanation.items()):
            min_val = min_values[i]
            max_val = max_values[i]
            low = values[0]
            high = values[1]

            new_low = (
                min_val
                if low == -1 * infinity
                else min_val + low * (max_val - min_val)
            )
            new_high = (
                max_val
                if high == infinity
                else min_val + high * (max_val - min_val)
            )

            # TODO round to 1 decimal place?
            denormalized_explanation["x{:d}".format(i)] = [
                round(new_low, 1),
                round(new_high, 1),
            ]

        return jsonify(explanation)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(port=port, debug=True)
