import { sideBar } from "./sideBar.js";
import { clamp_value, create_example, feature_dists_order, pretty_value, unscale } from "./utilities.js";
import { ExplanationTypes, OFFSET_UNSCALED, expl_colors, rect_values } from "./values.js";

export const liguisticDisplay = () => {
    let width;
    let height;
    let explanation;
    let dataset_details;
    let readable;
    let expl_type;

    const my = (selection) => {
        // extract explanation information
        const n_features = Object.keys(dataset_details["feature_names"]).length;
        const instance = explanation["instance"];
        const region = explanation["region"];
        // add a bounding rectangle
        const bbox_width = width - 20;
        const bbox_height = height - 20;
        const bbox_x = 10;
        const bbox_y = 10;
        const bbox_round = 3;
        const bbox_stroke = "black";
        const bbox_stroke_width = 0.5
        const bbox_stroke_opacity = 0.4
        selection.append("rect")
            .attr("id", "bbox")
            .attr("width", bbox_width)
            .attr("id", "bbox")
            .attr("height", bbox_height)
            .attr("fill", rect_values.fill)
            .attr("x", bbox_x)
            .attr("y", bbox_y)
            .attr("rx", bbox_round)
            .attr("ry", bbox_round)
            .attr("stroke", bbox_stroke)
            .attr("stroke-width", bbox_stroke_width)
            .attr("stroke-opacity", bbox_stroke_opacity)

        // ####################################### SIDEBAR CONTENT #######################################

        const sidebar_width_ratio = 0.33;
        const sidebar_margin = 10;
        const sidebar_width = bbox_width * sidebar_width_ratio;
        const sidebar_height = bbox_height - (sidebar_margin * 2);
        const sidebar_x = bbox_x + sidebar_margin;
        const sidebar_y = bbox_y + sidebar_margin;
        const sidebar = sideBar()
            .width(sidebar_width)
            .height(sidebar_height)
            .x(sidebar_x)
            .y(sidebar_y)
            .explanation(explanation)
            .dataset_details(dataset_details)
            .readable(readable);
        selection.call(sidebar);

        // ####################################### FUNCTION LIBRARY #######################################

        function get_feature_name(feature_i) {
            const feature_id = "x" + feature_i;
            const feature_name = dataset_details["feature_names"][feature_id];
            const pretty_feature_name = readable["pretty_feature_names"][feature_name];
            return pretty_feature_name
        }

        function get_feature_between(feature_i, region) {
            const feature_id = "x" + feature_i;
            const feature_name = dataset_details["feature_names"][feature_id]
            var lower_value = unscale(region[feature_id][0], feature_id, dataset_details);
            var upper_value = unscale(region[feature_id][1], feature_id, dataset_details);
            lower_value = clamp_value(lower_value, feature_id, readable, dataset_details);
            upper_value = clamp_value(upper_value, feature_id, readable, dataset_details);
            return [lower_value, upper_value]
        }

        const case_format = "allupper"
        function format_case(text) {
            if (case_format == "allupper") {
                return text.toUpperCase()
            }
            if (case_format == "alllower") {
                return text.toLowerCase()
            }
        }

        // ####################################### EXPLANATION BOX #######################################

        // add a containing box for the explanation
        const ebox_width_ratio = 1 - sidebar_width_ratio;
        const ebox_width = (bbox_width * ebox_width_ratio) - 3 * sidebar_margin;
        const ebox_height = bbox_height - (sidebar_margin * 2);
        const ebox_x = bbox_x + (sidebar_width_ratio * bbox_width) + (2 * sidebar_margin);
        const ebox_y = bbox_y + sidebar_margin;
        selection.append("rect")
            .attr("id", "ebox")
            .attr("width", ebox_width)
            .attr("height", ebox_height)
            .attr("fill", "white")
            .attr("x", ebox_x)
            .attr("y", ebox_y)
            .attr("rx", rect_values.round)
            .attr("ry", rect_values.round)
            .attr("stroke", rect_values.stroke)
            .attr("stroke-width", rect_values.stroke_width)
            .attr("stroke-linejoin", "round");

        // ####################################### BUILD EXPLANATION TEXT #######################################

        // compute the unscaled distance along each dimension and sort by it
        const [feature_distances, idx_order] = feature_dists_order(instance, region);

        // CREATE THE STARTING TEXT
        var expl_text = Array(); // lets build an array for templated text
        // ADD: Your <instance> would have been <desired outcome> rather than <undesired outcome> if your
        const start_text = "Your " + readable["scenario_terms"]["instance_name"].toLowerCase() + " would have been";
        expl_text.push([start_text, "black"]);
        expl_text.push([format_case(readable["scenario_terms"]["desired_outcome"]), expl_colors.desired]);
        expl_text.push(["rather than", "black"]);
        expl_text.push([format_case(readable["scenario_terms"]["undesired_outcome"]), expl_colors.undesired]);
        expl_text.push(["if your", "black"]);

        // ADD TEXT FOR ALTERED FEATURES
        var n_feats_listed = 0;
        for (let i = 0; i < n_features; i++) {
            // for features in order of largest change
            if (feature_distances[idx_order[i]] > 0) {
                const feature_id = "x" + idx_order[i];
                const feature_name = dataset_details["feature_names"][feature_id];
                if (n_feats_listed > 0) {
                    expl_text.push(["and your", "black"])
                }
                // ADD: <feature name> was between <good value low> and <good value high> rather than <bad value>
                expl_text.push([format_case(get_feature_name(idx_order[i])) + " was", "black"]);
                // get the instance and region values
                const feature_value = unscale(instance[feature_id], feature_id, dataset_details);
                const [lower_value, upper_value] = get_feature_between(idx_order[i], region);

                if (expl_type == ExplanationTypes.Region) {
                    expl_text.push(["between", "black"]);
                    // format the value text neatly
                    const lower_value_text = pretty_value(lower_value, feature_name, readable);
                    const upper_value_text = pretty_value(upper_value, feature_name, readable);
                    const range_text = lower_value_text + " and " + upper_value_text;
                    expl_text.push([range_text, expl_colors.desired]);
                }
                else if (expl_type == ExplanationTypes.Example) {
                    var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                    const example_val = create_example(feature_value, lower_value, upper_value, offset);
                    const example_text = pretty_value(example_val, feature_name, readable)
                    expl_text.push([example_text, expl_colors.desired]);
                }

                expl_text.push(["rather than", "black"])
                expl_text.push([pretty_value(feature_value, feature_name, readable), expl_colors.undesired]);
                n_feats_listed += 1; // number of altered features
            }
        }
        // if unaltered feature remain add the below
        if (n_feats_listed < n_features) {
            expl_text.push([", assuming all other features are the same", "black"]);
            // ADD TEXT FOR UNALTERED FEATURES
            var in_paren = false
            for (let i = 0; i < n_features; i++) {
                if (feature_distances[idx_order[i]] == 0) {
                    var feature_text = "";
                    if (!in_paren) {
                        feature_text += "(";
                        in_paren = true;
                    }
                    feature_text += format_case(get_feature_name(idx_order[i]))
                    const feature_id = "x" + idx_order[i];
                    const feature_name = dataset_details["feature_names"][feature_id];
                    const feature_value = unscale(instance[feature_id], feature_id, dataset_details);
                    expl_text.push([feature_text, "black"]);

                    var current_value_text;
                    if (expl_type == ExplanationTypes.Region) {
                        const [lower_value, upper_value] = get_feature_between(idx_order[i], region);
                        const lower_value_text = pretty_value(lower_value, feature_name, readable);
                        const upper_value_text = pretty_value(upper_value, feature_name, readable);
                        const range_text = lower_value_text + " and " + upper_value_text;
                        current_value_text = "between " + range_text;
                    }
                    else if (expl_type == ExplanationTypes.Example) {
                        current_value_text = pretty_value(feature_value, feature_name, readable);
                    }

                    // var current_value_text = get_feature_between(idx_order[i]);
                    if (i < (n_features - 1)) {  // if this is not the last feature add a semicolon
                        current_value_text += ";"
                    }
                    else { // if it is add a closing parenthesis
                        current_value_text += ")"
                    }
                    expl_text.push([current_value_text, "black"]);
                }
            }

        }
        // ADD TEXT FOR UNALTERED FEATURES
        // for (let i = 0; i < n_features; i++) {
        //     var feature_text = "";
        //     if (!in_paren) {
        //         feature_text += "(";
        //         in_paren = true;
        //     }
        //     feature_text += format_case(get_feature_name(idx_order[i]))
        //     const feature_id = "x" + idx_order[i];
        //     const feature_name = dataset_details["feature_names"][feature_id];
        //     const feature_value = unscale(instance[feature_id], feature_id, dataset_details);
        //     expl_text.push([feature_text, "black"]);

        //     var current_value_text;
        //     if (expl_type == ExplanationTypes.Region) {
        //         const [lower_value, upper_value] = get_feature_between(idx_order[i], region);
        //         const lower_value_text = pretty_value(lower_value, feature_name, readable);
        //         const upper_value_text = pretty_value(upper_value, feature_name, readable);
        //         const range_text = lower_value_text + " and " + upper_value_text;
        //         current_value_text = "between " + range_text;
        //     }
        //     else if (expl_type == ExplanationTypes.Example) {
        //         current_value_text = pretty_value(feature_value, feature_name, readable);
        //     }

        //     // var current_value_text = get_feature_between(idx_order[i]);
        //     if (i < (n_features - 1)) {  // if this is not the last feature add a semicolon
        //         current_value_text += ";"
        //     }
        //     else { // if it is add a closing parenthesis
        //         current_value_text += ")"
        //     }
        //     expl_text.push([current_value_text, "black"]);
        //     i += 1;
        // }

        // ####################################### RENDER EXPLANATION TEXT #######################################

        var combined_text = "";
        for (let i = 0; i < expl_text.length; i++) {
            combined_text += expl_text[i][0] + " "
        }

        // text box location
        const etext_margin = 30;
        const ebox_center_x = ebox_x + (ebox_width / 2)
        const etext_x = ebox_x + etext_margin;
        const etext_y = ebox_y + etext_margin + 40;
        const textbox = selection.append("text")
            .attr("x", etext_x)
            .attr("y", etext_y)
            .attr("fill", "black")
            .attr("class", "feature-details")
            .text(null);
        const n_expl_lines = render_textarray(textbox, expl_text, ebox_width - (2 * etext_margin));

        /** Given an array of [<text element>, <color>] tuples, render the svg text with the right color wrapped to the given width adapted from wrap (https://stackoverflow.com/questions/24784302/wrapping-text-in-d3) */
        function render_textarray(textbox, text_arr, width) {
            var lineNumber = 0;
            textbox.text(null); // clear the textbox
            var debug_val = 0;

            var entry;
            const x = textbox.attr("x");
            const y = textbox.attr("y");
            const dy = 0; //parseFloat(text.attr("dy")),
            const lineHeight = 1.8;
            var line_used_width = 0;
            var tspan;
            text_arr = text_arr.reverse()
            var entry_num = 0;
            const punctuation = [",", ".", "?", "!", ";", "'", "\""]
            while (entry = text_arr.pop()) { // for each entry in text_arr
                var entry_words = entry[0].split(/\s+/).reverse();
                // add a tspan of the right color, include a space betweent words but not punctuation
                if (line_used_width > 0 && !punctuation.includes(entry_words[entry_words.length - 1])) {
                    line_used_width += 5;
                }
                var tspan = textbox.append("tspan")
                    .attr("x", x)
                    .attr("y", y)
                    .attr("dy", lineNumber * lineHeight + dy + "em")
                    .attr("dx", line_used_width)
                    .attr("fill", entry[1]);
                var word;
                var line = [];
                var tspan_width = 0;
                while (word = entry_words.pop()) {
                    // add the words of the entry to this tspan
                    line.push(word);
                    tspan.text(line.join(" "));
                    // if that tspan gets wider than the acceptable width
                    tspan_width = tspan.node().getComputedTextLength();
                    if ((line_used_width + tspan_width) > width) {
                        line.pop();
                        tspan.text(line.join(" "));
                        // add a new one
                        lineNumber += 1
                        tspan = textbox.append("tspan")
                            .attr("x", x)
                            .attr("y", y)
                            .attr("dy", lineNumber * lineHeight + dy + "em")
                            .text(word)
                            .attr("fill", entry[1])
                        tspan_width = tspan.node().getComputedTextLength();
                        line_used_width = 0;
                        line = [word]
                    }
                }
                line_used_width += tspan_width;
            }
            return lineNumber;
        }
    }

    my.width = function (_) {
        return arguments.length
            ? ((width = +_), my)
            : width;
    };

    my.height = function (_) {
        return arguments.length
            ? ((height = +_), my)
            : height;
    };

    my.x = function (_) {
        return arguments.length
            ? ((x = +_), my)
            : x;
    };

    my.y = function (_) {
        return arguments.length
            ? ((y = +_), my)
            : y;
    };

    my.explanation = function (_) {
        return arguments.length ? ((explanation = _), my) : explanation;
    };

    my.dataset_details = function (_) {
        return arguments.length ? ((dataset_details = _), my) : dataset_details;
    };

    my.readable = function (_) {
        return arguments.length ? ((readable = _), my) : readable;
    };

    my.expl_type = function (_) {
        return arguments.length ? ((expl_type = _), my) : expl_type;
    };

    my.expl_type = function (_) {
        return arguments.length ? ((expl_type = _), my) : expl_type;
    };

    return my;
}