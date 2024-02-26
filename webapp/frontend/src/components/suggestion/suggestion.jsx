import { formatFeature, formatValue } from '@src/js/utilities.js';
import './suggestion.css';

export const Suggestion = ({ formatDict, featureDict, selectedInstance, explanations, currentExplanationIndex }) => {


    // Return the message for the suggestion box
    let suggestion_text = [
        ["Your application would have been", "text-black"],
        [formatDict["scenario_terms"]["desired_outcome"].toUpperCase(), "text-blue"],
        ["rather than", "text-black"],
        [formatDict["scenario_terms"]["undesired_outcome"].toUpperCase(), "text-red"],
        ["if your", "text-black"],
    ];

    let currentExplanation = explanations[currentExplanationIndex]

    if (currentExplanation === undefined) { // If the ranges are undefined
        return (
            <>
                <h2>Suggestion</h2>
                <i className="instructions-text">No suggestion available.</i>
            </>
        )
    }
    else {
        let nAlterations = 0;
        return (
            <div>
                <h2>Suggestion</h2>
                <div style={{ lineHeight: 1.3 }}>
                    {suggestion_text.map((entry, index) => (
                        <span key={index} className={entry[1]}>{entry[0] + " "}</span>
                    ))}

                    {Object.keys(featureDict).map((key, index) => {
                        let lower_bound = explanations[currentExplanationIndex][key][0];
                        let upper_bound = explanations[currentExplanationIndex][key][1];
                        let current_val = selectedInstance[key];

                        if (current_val < lower_bound || current_val > upper_bound) {
                            let alter_text = []
                            if (nAlterations > 0) {
                                alter_text.push(["and your"])
                            }
                            nAlterations = nAlterations + 1;
                            alter_text = alter_text.concat([
                                [formatFeature(key, formatDict), "text-black"],
                                ["was between", "text-black"],
                                [formatValue(lower_bound, key, formatDict), "text-blue"],
                                ["and", "text-black"],
                                [formatValue(upper_bound, key, formatDict), "text-blue"],
                                ["rather than", "text-black"],
                                [formatValue(current_val, key, formatDict), "text-red"],
                            ]);
                            return (<span key={index}>
                                {
                                    alter_text.map((entry, index) => (
                                        <span key={index} className={entry[1]}>{entry[0] + " "}</span>
                                    ))
                                }
                            </span>
                            )
                        }
                        else {
                            return;
                        }
                    })}
                    <span className="text-black"> assuming no other changes.</span>
                </div>

            </div >)
    }


};

export default Suggestion;
