import { clamp_value, create_example, pretty_value, unscale } from "./utilities.js";
import { ExplanationTypes, OFFSET_UNSCALED, expl_colors } from "./values.js";

export const TableTypes = Object.freeze({
    Application: "application",
    Explanation: "explanation"
})

export const explanationTable = () => {
    let width;
    let height;
    let x;
    let y;
    let explanation;
    let dataset_details;
    let readable;
    let expl_type;
    let idx_order;
    let feature_distances;
    let table_type;

    const my = (selection) => {
        // table formatting values
        const tr_stroke = "black"
        const tr_stroke_width = 2
        const tr_text_offset_x = width / 2
        const tr_text_offset_y = height / 2
        const table_row_offset = height
        const tr_pad = 10;
        const font_size = 16;

        const instance = explanation["instance"];
        const region = explanation["region"];
        const n_features = Object.keys(instance).length

        // add a header above the table
        if (table_type == "application") {
            var header_1 = readable["scenario_terms"]["instance_name"];
            var header_2 = readable["scenario_terms"]["undesired_outcome"];
            var header_2_color = expl_colors.undesired;
        } else if (table_type == "explanation") {
            var header_1 = "Explanation";
            var header_2 = readable["scenario_terms"]["desired_outcome"];
            var header_2_color = expl_colors.desired;
        }
        var etable_header = selection.append("text")
            .attr("x", x + width / 2)
            .attr("y", y - font_size / 2)
            .attr("class", "feature-details")
            .attr("class", "table-header")
            .attr("font-size", 16)
            .attr("font-style", "bold")
            .attr("fill", "black")
            .attr("text-anchor", "middle");
        etable_header.append("tspan")
            .text(header_1 + " (")
        etable_header.append("tspan")
            .text(header_2)
            .attr("fill", header_2_color);
        etable_header.append("tspan")
            .text(")")
            .attr("fill", "black");
        // create the table body
        for (let i = 0; i < n_features; i++) {
            // get the formatted feature name
            const feature_id = "x" + idx_order[i];
            const feature_name = dataset_details["feature_names"][feature_id];
            const has_change = feature_distances[idx_order[i]] > 0
            var pretty_feature_name = readable["pretty_feature_names"][feature_name];
            var value_text;
            var value_color;
            var label_color;

            if (table_type == "application") {
                // get the pretty feature value
                const feature_val = unscale(instance[feature_id], feature_id, dataset_details);
                value_text = pretty_value(feature_val, feature_name, readable);
                // color labels and values based on feature difference
                value_color = has_change ? expl_colors.altered_bad : expl_colors.unaltered
                label_color = has_change ? "black" : expl_colors.unaltered
            }
            else if (table_type == "explanation") {
                // get the upper and lower bound values
                var lower_val = unscale(region[feature_id][0], feature_id, dataset_details);
                var upper_val = unscale(region[feature_id][1], feature_id, dataset_details);
                lower_val = clamp_value(lower_val, feature_id, readable, dataset_details);
                upper_val = clamp_value(upper_val, feature_id, readable, dataset_details);

                if (expl_type == ExplanationTypes.Region) {
                    // format that text neatly
                    const lower_val_text = pretty_value(lower_val, feature_name, readable);
                    const upper_val_text = pretty_value(upper_val, feature_name, readable);
                    value_text = lower_val_text + " - " + upper_val_text;
                } else if (expl_type == ExplanationTypes.Example) {
                    const feature_value = unscale(instance[feature_id], feature_id, dataset_details);
                    var example_value = feature_value;
                    if (has_change) {
                        var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                        example_value = create_example(feature_value, lower_val, upper_val, offset);
                    }
                    value_text = pretty_value(example_value, feature_name, readable);
                }

                // choose label and value colors based on if the feature had to be changed
                value_color = has_change ? expl_colors.altered_good : expl_colors.unaltered
                label_color = has_change ? "black" : expl_colors.unaltered
            }

            // start a group for each table cell
            var tr = selection.append("g")
            // draw the cell box
            tr.append("rect")
                .attr("x", x)
                .attr("y", y + (table_row_offset * i))
                .attr("width", width)
                .attr("height", height)
                .attr("stroke", tr_stroke)
                .attr("stroke-width", tr_stroke_width);
            // add the feature label to the top-left of the box
            tr.append("text")
                .text(pretty_feature_name)
                .attr("x", x + tr_pad)
                .attr("y", y + (table_row_offset * i) + tr_pad + (font_size / 2))
                .attr("class", "feature-details")
                .attr("fill", "black")
                .attr("text-anchor", "start")
                .attr("fill", label_color)
                .attr("font-size", 16);
            // add the feature value to the center of the box
            tr.append("text")
                .text(value_text)
                .attr("x", x + tr_text_offset_x)
                .attr("y", y + (table_row_offset * i) + tr_text_offset_y + font_size / 2)
                .attr("class", "feature-details")
                .attr("fill", "black")
                .attr("text-anchor", "middle")
                .attr("fill", value_color)
                .attr("font-size", 16);
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

    my.idx_order = function (_) {
        return arguments.length ? ((idx_order = _), my) : idx_order;
    };

    my.feature_distances = function (_) {
        return arguments.length ? ((feature_distances = _), my) : feature_distances;
    };

    my.table_type = function (_) {
        return arguments.length ? ((table_type = _), my) : table_type;
    };

    my.expl_type = function (_) {
        return arguments.length ? ((expl_type = _), my) : expl_type;
    };

    return my;
}