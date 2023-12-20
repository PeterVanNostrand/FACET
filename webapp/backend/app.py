import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from webapp.app_utilities import run_facet, parse_dataset_info
from dataset import get_json_paths, DataInfo

# load app confiuration parameters
with open("./webapp/config.json", "r") as config_file:
    APP_CONFIG: dict = json.load(config_file)          # config file with app parameters
API_PORT: int = APP_CONFIG["API_PORT"]                 # specified port for RESTful explanation API
DS_NAME: str = APP_CONFIG["DATASET"]                   # the dataset we're explaining
DETAILS_PATH, HUMAN_PATH = get_json_paths(DS_NAME)     # the paths to the ds_details, and human_readible info
DS_INFO: DataInfo = None
HUMAN_FORMAT: dict = None

# configure the app
app = Flask(__name__)
CORS(app)
FACET_CORE = None               # the core facet system which generated explanations
SAMPLE_DATA: np.ndarray = None  # teh set of sample instances we populate for the demo


def init_app():
    global FACET_CORE, SAMPLE_DATA, DS_INFO, HUMAN_FORMAT

    print("\nApp initializing...\n")
    try:
        # initialize FACET (load data, train model, index explanations) and get samples
        FACET_CORE, SAMPLE_DATA = run_facet(ds_name=DS_NAME)
        # load the dataset info JSON file which is automatically generated by FACET
        DS_INFO = parse_dataset_info(DETAILS_PATH)
        # load the human readable JSON file used for display formatting
        with open(HUMAN_PATH, "r") as human_format_file:
            HUMAN_FORMAT = json.load(human_format_file)
        # append a mapping to FACET's col_ids x0, x1, ... , xN (from DS_INFO)
        HUMAN_FORMAT["feature_names"] = DS_INFO.col_names
    except Exception as e:
        print(f"ERROR: Failed to run FACET. Details:\n{e}")
        exit(1)
    print("\nApp initialized\n")


init_app()


@app.route("/facet/instances", methods=["GET"])
def get_test_instances():
    num_arrays, array_length = SAMPLE_DATA.shape
    json_data = []

    # Iterate over the arrays and build the dictionary
    samples = DS_INFO.unscale_points(SAMPLE_DATA)
    for instance in samples:
        instance_dict = DS_INFO.point_to_dict(instance)
        json_data.append(instance_dict)

    return jsonify(json_data)


@app.route("/facet/human_format", methods=["GET"])
def get_human_format():
    return jsonify(HUMAN_FORMAT)


@app.route("/facet/explanation", methods=["POST"])
def facet_explanation():
    try:
        data = request.json
        instance = DS_INFO.dict_to_point(data)
        instance = DS_INFO.scale_points(instance)
        print("input_data", instance)  # debug

        if len(instance.shape) == 1:  # if we only have one instance
            instance = instance.reshape(-1, instance.shape[0])

        # Perform explanations using manager explain
        explain_pred = FACET_CORE.predict(instance)
        # get the counterfactual points and regions from FACET
        points, regions = FACET_CORE.explain(instance, explain_pred)

        region = DS_INFO.unscale_rects(regions[0])
        region_dict = DS_INFO.rect_to_dict(region)

        return jsonify(region_dict)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/facet/explanation", methods=["GET"])
def get_weights():
    dummy_ordering = []
    try:
        current_weight = 0
        increment = APP_CONFIG["WEIGHT_INCREMENTS"]
        is_exponent = APP_CONFIG["WEIGHT_POWERS"]
        weights = []
    except Exception as e:
        return jsonify({"error": str(e)})

    try:
        for feature in dummy_ordering:
            if(feature["lock_state"]):
                weights.append(1e-7) #value should be very close to 0
            elif(is_exponent):
                weights.append(pow(current_weight, increment))
                current_weight = weights[-1]
            else:
                weights.append(current_weight + increment)
                current_weight += increment

            return jsonify({"weights":np.vectorize(weights)})
    except Exception as e:
        return jsonify({"error": str(e)})
    
    


if __name__ == "__main__":
    app.run(port=API_PORT, debug=True)
