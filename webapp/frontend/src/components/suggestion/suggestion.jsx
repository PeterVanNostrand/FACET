import '../../css/suggestion.css';
import { formatFeature, formatValue } from '../../js/utilities';

const Suggestion = ({formatDict, selectedInstance, explanations, currentExplanationIndex}) => {

    // Make a singular suggest to be incorpoated into the full suggestion message
    const make_singluar_suggestion = (suggestion) => {

        // your FEATURE was between LOWER BOUND and UPPER BOUND rather than CURENT-VALUE
        let inline_suggestion = "your <span class=\"span-blue\">" + formatFeature(suggestion[0], formatDict) + "</span> was betweem <span class=\"span-blue\">" + formatValue(suggestion[2], suggestion[0], formatDict) + " and " + formatValue(suggestion[3], suggestion[0], formatDict) + "</span> rather than <span class=\"span-red\">" + formatValue(suggestion[1], suggestion[0], formatDict) + "</span>"

        return inline_suggestion
    }

    // Conscutrct the sub suggestion message for when the application rejected
    const construct_suggestion = (suggestions) => {
        // Start of the message
        let message_start = '<div class="div-1"><span class=\"span-black\">Your application would have been <span class="span-blue"> APPROVED</span> rather than <span class="span-red"> REJECTED</span> if'
        // Middle of the message
        let message_middle = ''
        // End of the message
        let message_end = ', assuming no other changes.</span></div>'

        // If there is only a single attribute suggestion
        if (suggestions.lenth == 1){
            // Set the middle message to that
            message_middle = make_singluar_suggestion(suggestions[0])
        } else {
            // For each suggestion
            for (let i = 0; i < suggestions.length; i++){
                // If this suggestion is the final one
                if (i == suggestions.length - 1){
                    // add an "and" before it
                    message_middle = message_middle.concat(" and")

                    // if this suggest is not the finla or first one
                } else if (i != 0) {
                    // add a "," before it
                    message_middle = message_middle.concat(",")
                }

                // Add the sinular suggestion to the middle message
                message_middle = message_middle.concat(" " + make_singluar_suggestion(suggestions[i]))
            }
        }

        // Construct the final message: Start + Middle + End
        let final_message = message_start.concat(" " + message_middle + message_end)

        // Return message
        return final_message
    }


    // Return a message for the suggestion element
    const construct_message = (suggestions) => {
        // Content of the message
        let content

        // If the suggesitons are currently undefined
        if (suggestions === undefined){
            // Set content to a loading message
            content = <div className="div-1">Loading suggestion...</div>

            // If there are no suggestions
        } else if (suggestions.length == 0){
            // Set content to an accepted message
            content = <div className="div-1">Congrats! Your application was 
            <span className="span-blue"> accepted</span></div>

        } else {
            // Otherwise, construct a suggestion message
            let sub_message = construct_suggestion(suggestions)
            // Write the html from sub-message and set innerhtml
            content = <div className="div-1" dangerouslySetInnerHTML={{ __html: sub_message}}></div>
        }

        // Return the html element for suggestion box
        return <div className="colored-box">
                <div>
                     <span className="span-2">Suggestion</span>
                        {content} </div></div>
    }

    // Attribute keys
    let keys = Object.keys(selectedInstance)

    // Get the allowed ranges for the attributes in the selected explanation
    let explanation_attribute_ranges = explanations[currentExplanationIndex]

    // If the ranges are undefined
    if (explanation_attribute_ranges === undefined){
        // Return an undefined message for the suggestion box
        return construct_message(undefined)
    }

    let attribute_suggestions = []

    // For each of the attributes
    for (let i = 0; i < keys.length; i++){
        // Get the Attribute's currnet value
        let current = selectedInstance[keys[i]]
        // Get the Attribute's upper bound
        let upper_bound = explanation_attribute_ranges[keys[i]][1]
        // Get the Attribute's lower bound
        let lower_bound = explanation_attribute_ranges[keys[i]][0]

        // If the current value is within the attribute's allowed range
        if (current <= upper_bound && current >= lower_bound){
            // Do nothing; no suggestion needs to be made
        } else {
            //Otherwise, calculate the closest bound to the attribute's currnet value
            let closest_bound = Math.abs(current - upper_bound) < Math.abs(current - lower_bound) ? upper_bound : lower_bound
            // (Not needed in the current suggestion message; but good to have if need be)
        
            // Add a suggestion to attribute_suggetion
            //   attribute-ID, current-value, lower-bound, upper-bound, closest-bound
            attribute_suggestions.push([keys[i], current, lower_bound, upper_bound, closest_bound])
        }
    }

    // Return the message for the suggestion box
    return construct_message(attribute_suggestions)    
}; 

export default Suggestion;
