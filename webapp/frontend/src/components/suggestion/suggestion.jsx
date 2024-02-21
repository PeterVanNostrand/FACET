import React, { useEffect, useState } from 'react';
import '../../css/suggestion.css';
import { formatFeature, formatValue } from '../../js/utilities';

const Suggestion = ({formatDict, selectedInstance, explanations, currentExplanationIndex}) => {

    const make_singluar_suggestion = (suggestion) => {
        // your FEATUER was ACCEPTED_VALUE rather then CURRENT-VALUE

        console.log(suggestion)
        console.log(suggestion[0])

        let inline_suggestion = "your <span class=\"span-blue\">" + formatFeature(suggestion[0], formatDict) + "</span> was betweem <span class=\"span-blue\">" + formatValue(suggestion[2], suggestion[0], formatDict) + " and " + formatValue(suggestion[3], suggestion[0], formatDict) + "</span> rather than <span class=\"span-red\">" + formatValue(suggestion[1], suggestion[0], formatDict) + "</span>"

        return inline_suggestion
    }

    const constuct_sub_message = (suggestions) => {
        let message_start = '<div class="div-1"><span class=\"span-black\">Your application would have been <span class="span-blue"> APPROVED</span> rather than <span class="span-red"> REJECTED</span> if'
        let message_middle = ''
        let message_end = ', assuming no other changes.</span></div>'

        if (suggestions.lenth == 1){
            message_middle = make_singluar_suggestion(suggestions[0])
        } else {
            for (let i = 0; i < suggestions.length; i++){
                if (i == suggestions.length - 1){
                    message_middle = message_middle.concat(" and")
                } else if (i != 0) {
                    message_middle = message_middle.concat(",")
                }

                message_middle = message_middle.concat(" " + make_singluar_suggestion(suggestions[i]))
            }
        }

        let final_message = message_start.concat(" " + message_middle + message_end)

        return final_message
    }



    const construct_full_message = (suggestions) => {
        let content

        if (suggestions === undefined){
            content = <div className="div-1">Loading suggestion...</div>
        } else if (suggestions.length == 0){
            content = <div className="div-1">Congrats! Your application was 
            <span className="span-blue"> accepted</span></div>
        } else {
            let sub_message = constuct_sub_message(suggestions)
            content = <div className="div-1" dangerouslySetInnerHTML={{ __html: sub_message}}></div>
           // content.innerHTML = sub_message

            
            // content =  (<div className="div-1">
            //     {constuct_sub_message(suggestions)}
            //     </div>)
        }

        console.log(content)

        return <div className="colored-box">
                <div>
                     <span className="span-2">Suggestion</span>
                        {content} </div></div>
    }

    let keys = Object.keys(selectedInstance)

    let ranges = explanations[currentExplanationIndex]

    if (ranges === undefined){
        return construct_full_message(undefined)
    }

    let attributes = []

    for (let i = 0; i < keys.length; i++){
        let current = selectedInstance[keys[i]]
        let upper_bound = ranges[keys[i]][1]
        let lower_bound = ranges[keys[i]][0]

        if (current <= upper_bound && current >= lower_bound){
            // nothing, the value is within the range
        } else {
            let closest_bound = Math.abs(current - upper_bound) < Math.abs(current - lower_bound) ? upper_bound : lower_bound

            attributes.push([keys[i], current, lower_bound, upper_bound, closest_bound])
        }
    }

    console.log(attributes)

    return construct_full_message(attributes)


    
    //return <div> <p>hello world <span style={{"color":"blue"}}>blue</span></p></div>

    
}; 

export default Suggestion;

    // // compute the unscaled distance along each dimension and sort by it
// const [feature_distances, idx_order] = feature_dists_order(instance, region);

// // CREATE THE STARTING TEXT
// var expl_text = Array(); // lets build an array for templated text
// // ADD: Your <instance> would have been <desired outcome> rather than <undesired outcome> if your
// const start_text = "Your " + readable["scenario_terms"]["instance_name"].toLowerCase() + " would have been";
// expl_text.push([start_text, "black"]);
// expl_text.push([format_case(readable["scenario_terms"]["desired_outcome"]), expl_colors.desired]);
// expl_text.push(["rather than", "black"]);
// expl_text.push([format_case(readable["scenario_terms"]["undesired_outcome"]), expl_colors.undesired]);
// expl_text.push(["if your", "black"]);

// // ADD TEXT FOR ALTERED FEATURES
// var n_feats_listed = 0;
// for (let i = 0; i < n_features; i++) {
//     // for features in order of largest change
//     if (feature_distances[idx_order[i]] > 0) {
//         const feature_id = "x" + idx_order[i];
//         const feature_name = dataset_details["feature_names"][feature_id];
//         if (n_feats_listed > 0) {
//             expl_text.push(["and your", "black"])
//         }
//         // ADD: <feature name> was between <good value low> and <good value high> rather than <bad value>
//         expl_text.push([format_case(get_feature_name(idx_order[i])) + " was", "black"]);
//         // get the instance and region values
//         const feature_value = unscale(instance[feature_id], feature_id, dataset_details);
//         const [lower_value, upper_value] = get_feature_between(idx_order[i], region);

//         if (expl_type == ExplanationTypes.Region) {
//             expl_text.push(["between", "black"]);
//             // format the value text neatly
//             const lower_value_text = pretty_value(lower_value, feature_name, readable);
//             const upper_value_text = pretty_value(upper_value, feature_name, readable);
//             const range_text = lower_value_text + " and " + upper_value_text;
//             expl_text.push([range_text, expl_colors.desired]);
//         }
//         else if (expl_type == ExplanationTypes.Example) {
//             var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
//             const example_val = create_example(feature_value, lower_value, upper_value, offset);
//             const example_text = pretty_value(example_val, feature_name, readable)
//             expl_text.push([example_text, expl_colors.desired]);
//         }

//         expl_text.push(["rather than", "black"])
//         expl_text.push([pretty_value(feature_value, feature_name, readable), expl_colors.undesired]);
//         n_feats_listed += 1; // number of altered features
//     }
// }
// // if unaltered feature remain add the below
// if (n_feats_listed < n_features) {
//     expl_text.push([", assuming all other features are the same", "black"]);
//     // ADD TEXT FOR UNALTERED FEATURES
//     var in_paren = false
//     for (let i = 0; i < n_features; i++) {
//         if (feature_distances[idx_order[i]] == 0) {
//             var feature_text = "";
//             if (!in_paren) {
//                 feature_text += "(";
//                 in_paren = true;
//             }
//             feature_text += format_case(get_feature_name(idx_order[i]))
//             const feature_id = "x" + idx_order[i];
//             const feature_name = dataset_details["feature_names"][feature_id];
//             const feature_value = unscale(instance[feature_id], feature_id, dataset_details);
//             expl_text.push([feature_text, "black"]);

//             var current_value_text;
//             if (expl_type == ExplanationTypes.Region) {
//                 const [lower_value, upper_value] = get_feature_between(idx_order[i], region);
//                 const lower_value_text = pretty_value(lower_value, feature_name, readable);
//                 const upper_value_text = pretty_value(upper_value, feature_name, readable);
//                 const range_text = lower_value_text + " and " + upper_value_text;
//                 current_value_text = "between " + range_text;
//             }
//             else if (expl_type == ExplanationTypes.Example) {
//                 current_value_text = pretty_value(feature_value, feature_name, readable);
//             }

//             // var current_value_text = get_feature_between(idx_order[i]);
//             if (i < (n_features - 1)) {  // if this is not the last feature add a semicolon
//                 current_value_text += ";"
//             }
//             else { // if it is add a closing parenthesis
//                 current_value_text += ")"
//             }
//             expl_text.push([current_value_text, "black"]);
//         }
//     }

// }