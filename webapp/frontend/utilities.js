// utilities.js
// Frontend Utility Functions

/**
 * Takes in either a pascal case or snake case string and formats it with spaces
 * @param {String} str - Input string to format
 * @returns {String} - formatted string. (ex: BananaPudding_In_Salami -> Banana Pudding In Salami)
 */
export function formatString(str){
    const regex = /[A-Z][a-z]*/g;
    const matchArray = [...str.matchAll(regex)];
    return(matchArray.join(' '));
}

/**
 * Takes in some float/int and formats it to be more human readable (Thanks Peter!)
 * @param {String} featureValue - Input number as string
 * @param {String} featureName - Input name of the feature
 * @param {Object} readable - Input dictionary of units
 * @returns {String} - formatted string. (ex: BananaPudding_In_Salami -> Banana Pudding In Salami)
 */
export function formatNumerical(featureValue, featureName, readable) {
    var valueText = "";
    var trimmedValue = parseFloat(featureValue).toFixed(readable["feature_decimals"][featureName])
    if (readable["feature_units"][featureName] == "$") {
        valueText += formatter.format(trimmedValue);
    }
    else {
        valueText += trimmedValue + " " + readable["feature_units"][featureName];
    }
    return valueText
}
  // Export the functions if using in a Node.js environment
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatString,
        formatNumerical
    };
  }