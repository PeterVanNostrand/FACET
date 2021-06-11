from heead import HEEAD
from dataset import load_data
from utilities.metrics import conversion_rate
from utilities.metrics import classification_metrics
from utilities.metrics import mean_distance
import pandas as pd
import os


def check_create_directory(results_dir="./results"):
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)


def vary_difference():
    '''
    Experiment to observe the effect of the difference value using in decision tree explanation
    '''
    # Load the dataset
    x, y = load_data("thyroid", normalize=True)

    # dataframe to store results of all experimental runs
    results = pd.DataFrame(columns=["difference", "accuracy", "precision",
                           "recall", "f1", "conversion_rate", "mean_distance"])
    check_create_directory()

    differences = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for diff in differences:
        # Create, train, and predict with the model
        params = {
            "rf_difference": diff,
            "rf_distance": "Euclidean",
            "rf_k": 1
        }
        model = HEEAD(detectors=["RandomForest"], aggregator="LogisticRegression",
                      explainer="BestCandidate", hyperparameters=params)
        model.train(x, y)
        preds = model.predict(x)

        # generate the explanations
        explanations = model.explain(x, y)

        # measure model performance
        accuracy, precision, recall, f1 = classification_metrics(preds, y, verbose=False)
        conv_rate = conversion_rate(explanations)
        mean_dist = mean_distance(x, explanations)

        # save the performance
        run_result = {
            "difference": diff,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conversion_rate": conv_rate,
            "mean_distance": mean_dist
        }
        results = results.append(run_result, ignore_index=True)

    # save the results
    results.to_csv("./results/vary_difference.csv")
