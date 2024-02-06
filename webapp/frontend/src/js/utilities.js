// utilities.js
// Frontend Utility Functions

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