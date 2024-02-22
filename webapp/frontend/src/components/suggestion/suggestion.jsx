import { formatFeature, formatValue } from '@src/js/utilities.js';
import './suggestion.css';

export const Suggestion = ({ formatDict, featureDict, selectedInstance, explanations, currentExplanationIndex }) => {

    // Make a singular suggest to be incorpoated into the full suggestion message
    // const make_singluar_suggestion = (suggestion) => {

    //     // your FEATURE was between LOWER BOUND and UPPER BOUND rather than CURENT-VALUE
    //     let inline_suggestion = "your <span class=\"span-blue\">" + formatFeature(suggestion[0], formatDict) + "</span> was betweem <span class=\"span-blue\">" + formatValue(suggestion[2], suggestion[0], formatDict) + " and " + formatValue(suggestion[3], suggestion[0], formatDict) + "</span> rather than <span class=\"span-red\">" + formatValue(suggestion[1], suggestion[0], formatDict) + "</span>"

    //     return inline_suggestion
    // }


    // Conscutrct the sub suggestion message for when the application rejected
    // const construct_suggestion = (suggestions) => {
    //     // Start of the message

    //     // Middle of the message
    //     let message_middle = ''
    //     // End of the message
    //     let message_end = ', assuming no other changes.</span>'

    //     // If there is only a single attribute suggestion
    //     if (suggestions.lenth == 1) {
    //         // Set the middle message to that
    //         message_middle = make_singluar_suggestion(suggestions[0])
    //     } else {
    //         // For each suggestion
    //         for (let i = 0; i < suggestions.length; i++) {
    //             // If this suggestion is the final one
    //             if (i == suggestions.length - 1) {
    //                 // add an "and" before it
    //                 message_middle = message_middle.concat(" and")

    //                 // if this suggest is not the finla or first one
    //             } else if (i != 0) {
    //                 // add a "," before it
    //                 message_middle = message_middle.concat(",")
    //             }

    //             // Add the sinular suggestion to the middle message
    //             message_middle = message_middle.concat(" " + make_singluar_suggestion(suggestions[i]))
    //         }
    //     }

    // Construct the final message: Start + Middle + End
    // let final_message = message_start.concat(" " + message_middle + message_end)

    // Return message
    // return final_message
    // }


    // Return a message for the suggestion element
    // const construct_message = (suggestions) => {

    //     let content // Content of the message
    //     if (suggestions === undefined) { // If the suggesitons are currently undefined
    //         // Set content to a loading message
    //         content = <p>Loading suggestion...</p>
    //     } else if (suggestions.length == 0) { // If there are no suggestions
    //         // Set content to an accepted message
    //         content = <p>No suggestion availible</p>
    //     } else { // Otherwise, construct a suggestion message
    //         let sub_message = construct_suggestion(suggestions)
    //         // Write the html from sub-message and set innerhtml
    //         content = <div dangerouslySetInnerHTML={{ __html: sub_message }}></div>
    //     }

    //     // Return the html element for suggestion box
    //     return <div>
    //         <div>
    //             <h2>Suggestion</h2>
    //             {content}
    //         </div>
    //     </div>
    // }

    // Attribute keys
    // let keys = Object.keys(selectedInstance)

    // // Get the allowed ranges for the attributes in the selected explanation
    // let explanation_attribute_ranges = explanations[currentExplanationIndex]

    // // If the ranges are undefined
    // if (explanation_attribute_ranges === undefined) {
    //     // Return an undefined message for the suggestion box
    //     return construct_message(undefined)
    // }

    // let attribute_suggestions = []

    // // For each of the attributes
    // for (let i = 0; i < keys.length; i++) {
    //     // Get the Attribute's currnet value
    //     let current = selectedInstance[keys[i]]
    //     // Get the Attribute's upper bound
    //     let upper_bound = explanation_attribute_ranges[keys[i]][1]
    //     // Get the Attribute's lower bound
    //     let lower_bound = explanation_attribute_ranges[keys[i]][0]

    //     // If the current value is within the attribute's allowed range
    //     if (current <= upper_bound && current >= lower_bound) {
    //         // Do nothing; no suggestion needs to be made
    //     } else {
    //         //Otherwise, calculate the closest bound to the attribute's currnet value
    //         let closest_bound = Math.abs(current - upper_bound) < Math.abs(current - lower_bound) ? upper_bound : lower_bound
    //         // (Not needed in the current suggestion message; but good to have if need be)
    //         // Add a suggestion to attribute_suggetion
    //         //   attribute-ID, current-value, lower-bound, upper-bound, closest-bound
    //         attribute_suggestions.push([keys[i], current, lower_bound, upper_bound, closest_bound])
    //     }
    // }

    // Return the message for the suggestion box
    let suggestion_text = [
        ["Your application would have been", "text-black"],
        [formatDict["scenario_terms"]["desired_outcome"].toUpperCase(), "text-blue"],
        ["rather than", "text-black"],
        [formatDict["scenario_terms"]["undesired_outcome"].toUpperCase(), "text-red"],
        ["if your", "text-black"],
    ];

    // let suggestions = attribute_suggestions
    // for (let i = 0; i < suggestions.length; i++) {
    //     if (i == suggestions.length - 1) { // If this suggestion is the final one add an "and" before it
    //         suggestion_text.push([" and", "text-black"])
    //     } else if (i != 0) { // if this suggest is not the final or first one add a "," before it
    //         suggestion_text.push([",", "text-black"])
    //     }

    //     suggestion_text.push[["your", "text-black"]];
    //     let suggestion = suggestions[i]
    //     // console.log(formatFeature(suggestion[0], formatDict))
    //     suggestion_text.push[[formatFeature(suggestion[0], formatDict), "text-black"]];
    //     // suggestion_text.push[["was between", "text-black"]];
    //     // suggestion_text.push[[formatValue(suggestion[2], suggestion[0], formatDict), "text-blue"]];
    //     // suggestion_text.push[["and", "text-black"]];
    //     // suggestion_text.push[[formatValue(suggestion[3], suggestion[0], formatDict), "text-blue"]];
    //     // suggestion_text.push[["rather than", "text-black"]];
    //     // suggestion_text.push[[formatValue(suggestion[1], suggestion[0], formatDict), "text-red"]];
    //     // let inline_suggestion = "your <span class=\"span-blue\">" +  + "</span> was betweem <span class=\"span-blue\">" +  + " and " + formatValue(suggestion[3], suggestion[0], formatDict) + "</span> rather than <span class=\"span-red\">" +  + "</span>"
    //     // Add the sinular suggestion to the middle message
    //     // message_middle = message_middle.concat(" " + make_singluar_suggestion(suggestions[i]))
    // }

    // {
    //     Object.keys(formatDict).map((key, index) => (
    //         suggestion_text.push([{ index }, "text-black"])
    //     ))
    // }

    let currentExplanation = explanations[currentExplanationIndex]

    if (currentExplanation === undefined) { // If the ranges are undefined
        return <i className="instructions-text">No suggestion available.</i>
    }
    else {
        let nAlterations = 0;
        return (<div>
            <h2>Suggestion</h2>
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
        </div >)
    }


};

export default Suggestion;
