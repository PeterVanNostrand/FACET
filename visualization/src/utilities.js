import { CURRENCY_DIGITS, ExampleGeneration, FEATURE_ORDER, FeatureOrders, GENERATION_TYPE } from "./values";

export function wrap(text, width) {
    // function for creating multi line wrapped text taken from the below
    // https://stackoverflow.com/questions/24784302/wrapping-text-in-d3
    var nlines = 1;
    text.each(function () {
        var text = d3.select(this),
            words = text.text().split(/\s+/).reverse(),
            word,
            line = [],
            lineNumber = 0,
            lineHeight = 1.5, // ems
            x = text.attr("x"),
            y = text.attr("y"),
            dy = 0, //parseFloat(text.attr("dy")),
            tspan = text.text(null)
                .append("tspan")
                .attr("x", x)
                .attr("y", y)
                .attr("dy", dy + "em");
        while (word = words.pop()) {
            line.push(word);
            tspan.text(line.join(" "));
            if (tspan.node().getComputedTextLength() > width) {
                line.pop();
                tspan.text(line.join(" "));
                line = [word];
                nlines += 1
                tspan = text.append("tspan")
                    .attr("x", x)
                    .attr("y", y)
                    .attr("dy", ++lineNumber * lineHeight + dy + "em")
                    .text(word)
            }
        }
    });
    return nlines;
}

export function unscale(scaled_value, feature_id, dataset_details) {
    var feature_value = scaled_value
    if (dataset_details["normalized"]) {
        var feature_range = dataset_details["max_values"][feature_id] - dataset_details["min_values"][feature_id]
        var feature_value = (feature_value * feature_range) + dataset_details["min_values"][feature_id];
    }
    return feature_value
}

const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: CURRENCY_DIGITS,
    maximumFractionDigits: CURRENCY_DIGITS,
});

/**
 * 
 * @param {*} feature_value the numeric value of the feature, e.g., 1000
 * @param {*} feature_name  the name of the feature, e.g., Applicant Income
 * @param {*} readable      a human readable info dictionary containing the number of units and decimals to use
 * @returns 
 */
export function pretty_value(feature_value, feature_name, readable) {
    var value_text = "";
    var trimmed_value = parseFloat(feature_value).toFixed(readable["feature_decimals"][feature_name])
    if (readable["feature_units"][feature_name] == "$") {
        value_text += formatter.format(trimmed_value);
    }
    else {
        value_text += trimmed_value + " " + readable["feature_units"][feature_name];
    }
    return value_text
}

export function argsort(arr) {
    var len = arr.length;
    var indices = new Array(len);
    for (var i = 0; i < len; ++i) indices[i] = i;
    indices.sort(function (a, b) { return arr[a] < arr[b] ? -1 : arr[a] > arr[b] ? 1 : 0; });
    return indices;
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

export function create_example(instance_val, lower_val, upper_val, offset) {
    var example_val;
    var region_half_width = (upper_val - lower_val) / 2;
    if (region_half_width < offset || GENERATION_TYPE == ExampleGeneration.Centered) {
        offset = region_half_width; // use half the range width instead
    }

    // if the value is too low
    if (instance_val <= lower_val) {
        example_val = lower_val + offset; // increase it to fall in the range
    }
    // if the value is too hight
    else if (instance_val >= upper_val) {
        example_val = upper_val - offset // decrease it to fall in the range
    }

    return example_val;
}

/* Clamps the given value to the semantic min/max value for that feature. If no meaningful semantic min/max value is availible it uses the dataset min/max values*/
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