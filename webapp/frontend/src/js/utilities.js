// utilities.js
// Frontend Utility Functions

export const FeatureOrders = Object.freeze({
    Dataset: "dataset",
    LargestChange: "largestchange"
})

export const ExplanationTypes = Object.freeze({
    Region: "region",
    Example: "example"
})

export const ExampleGeneration = Object.freeze({
    Nearest: "nearest",
    Centered: "centered"
})

export const FEATURE_ORDER = FeatureOrders.Dataset
export const GENERATION_TYPE = ExampleGeneration.Nearest;

/**
 * Takes in some float/int and formats it to be more human readable (Thanks Peter!)
 * @param {String} featureValue - Input number as string, e.g. 123.4567
 * @param {String} colID - Input name of the feature, e.g. x1, x2, ... xN
 * @param {Object} formatDict - Input dictionary of units
 * @returns {String} valueText - formatted string. $123.46
 */
export function formatValue(featureValue, colID, formatDict) {
    const colName = formatDict["feature_names"][colID]
    var valueText = "";
    var trimmed_value = parseFloat(featureValue).toFixed(formatDict["feature_decimals"][colName])

    if (formatDict["feature_units"][colName] == "$") {
        const formatter = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: formatDict["feature_decimals"][colName],
            maximumFractionDigits: formatDict["feature_decimals"][colName],
        });
        valueText += formatter.format(trimmed_value);
    }
    else {
        valueText += trimmed_value + " " + formatDict["feature_units"][colName];
    }
    return valueText
}


/**
 * Takes in some column id and returns the pretty feature
  * @param {String} colID - Input name of the feature, e.g. x1, x2, ... xN
 * @param {Object} formatDict - Input dictionary of units
 * @returns {String} featureText - formatted string. $123.46
 */
export function formatFeature(colID, formatDict) {
    const rawFeatureName = formatDict["feature_names"][colID];
    const featureText = formatDict["pretty_feature_names"][rawFeatureName]
    return featureText
}

/**
 * Clamps the given value to the semantic min/max value for that feature. If no meaningful semantic min/max value is availible it uses the dataset min/max values
 * @param {*} value             numeric value, e.g., 1000
 * @param {*} feature_id        feature id, e.g., Applicant Income
 * @param {*} readable          the human readable info JSON object with semantic min/max
 * @param {*} dataset_details   the dataset details JSON object with observed min/max
 * @returns 
 */
export function clamp_value(value, feature_id, readable, dataset_details) {
    var clampped_value = value;
    const feature_name = dataset_details["feature_names"][feature_id];

    // clamp to the low end
    var feature_min = readable["semantic_min"][feature_name];
    if (!feature_min) { feature_min = dataset_details["min_values"][feature_id]; }
    clampped_value = Math.max(clampped_value, feature_min);

    // clamp to the high end
    var feature_max = readable["semantic_max"][feature_name]; // use the semantic min
    if (!feature_max) { feature_max = dataset_details["max_values"][feature_id]; }
    clampped_value = Math.min(clampped_value, feature_max);

    return clampped_value;
}



/** A function which computes and returns the size of change along each feature and a list of indices for largest to smallest change order */
export function feature_dists_order(instance, region) {
    const n_features = Object.keys(instance).length;
    var feature_distances = Array(n_features).fill(0)
    for (let i = 0; i < n_features; i++) {
        var feature_id = "x" + i
        var feature_value = instance[feature_id]
        var lower_bound = parseFloat(region[feature_id][0])
        var upper_bound = parseFloat(region[feature_id][1])

        if (feature_value < lower_bound) {
            feature_distances[i] = (lower_bound - feature_value)
        } else if (feature_value > upper_bound) {
            feature_distances[i] = (feature_value - upper_bound)
        }
    }
    var idx_order;
    if (FEATURE_ORDER == FeatureOrders.Dataset) {
        idx_order = Array.from(Array(n_features).keys())
    } else if (FEATURE_ORDER == FeatureOrders.LargestChange) {
        idx_order = argsort(feature_distances).reverse();
    }
    return [feature_distances, idx_order];
}