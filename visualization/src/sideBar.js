import { pretty_value, unscale, wrap } from "./utilities.js";
import { rect_values } from "./values.js";

export const sideBar = () => {
    let width;
    let height;
    let x;
    let y;
    let explanation;
    let dataset_details;
    let readable;

    const my = (selection) => {
        // add a sidebar box for details
        selection.append("rect")
            .attr("id", "sidebar")
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "white")
            .attr("x", x)
            .attr("y", y)
            .attr("rx", rect_values.round)
            .attr("ry", rect_values.round)
            .attr("stroke", rect_values.stroke)
            .attr("stroke-width", rect_values.stroke_width)
            .attr("stroke-linejoin", "round");

        // STATUS label
        const text_margin_left = 15;
        const text_margin_top = 40;
        selection.append("text")
            .text("STATUS")
            .attr("class", "header-text")
            .attr("x", x + text_margin_left)
            .attr("y", y + text_margin_top)
            .attr("fill", "black")
            .attr("font-size", 20);

        // dividing line properties
        const line_width = 2;
        const line_text_expand = 5;
        const line_length = width - (text_margin_left * 2) + line_text_expand;
        const line_x = x + text_margin_left - line_text_expand

        // STATUS line
        const status_line_y = y + text_margin_top + 10
        selection.append("line")
            .attr("x1", line_x)
            .attr("x2", line_x + line_length)
            .attr("y1", status_line_y)
            .attr("y2", status_line_y)
            .attr("stroke", rect_values.stroke)
            .attr("stroke-width", line_width);

        // status message
        const center = x + text_margin_left + (width - 2 * text_margin_left) / 2
        const reject_text = selection.append("text")
            .attr("x", center)
            .attr("y", status_line_y + 30)
            .attr("fill", "black")
            .text("Our algorithm has decided that your loan application should be")
            .attr("text-anchor", "middle")
            .attr("class", "feature-details");
        const n_reject_lines = wrap(reject_text, (width - 2 * text_margin_left))

        const predicted_status_y = status_line_y + n_reject_lines * 45
        selection.append("text")
            .text(readable["scenario_terms"]["undesired_outcome"].toUpperCase())
            .attr("x", center)
            .attr("y", predicted_status_y)
            .attr("id", "predicated-status")
            .attr("text-anchor", "middle")

        // APPLICATION label
        const application_label_y = predicted_status_y + text_margin_top
        selection.append("text")
            .text(readable["scenario_terms"]["instance_name"].toUpperCase())
            .attr("class", "header-text")
            .attr("x", x + text_margin_left)
            .attr("y", application_label_y)
            .attr("fill", "black")
            .attr("font-size", 20);

        // APPLICATION line
        const features_line_y = application_label_y + 10
        selection.append("line")
            .attr("x1", line_x)
            .attr("x2", line_x + line_length)
            .attr("y1", features_line_y)
            .attr("y2", features_line_y)
            .attr("stroke", rect_values.stroke)
            .attr("stroke-width", line_width);

        // APPLICATION FEATURE VALUE CONTENT
        const text_offset = 25;
        const instance = explanation["instance"];
        const n_features = Object.keys(instance).length
        for (let i = 0; i < n_features; i++) {
            // get the raw feature name and its formatted pretty equivalent
            var feature_id = "x" + i // x0, x1, ...., xn
            var feature_name = dataset_details["feature_names"][feature_id]; // raw name e.g. applicant_income
            var pretty_name = readable["pretty_feature_names"][feature_name]; // formatted name e.g. Applicant Name

            // get the the raw feature value and unscale it if needed
            var feature_value = unscale(instance[feature_id], feature_id, dataset_details)

            // format the value to a string and add unit signs
            var value_text = pretty_value(feature_value, dataset_details["feature_names"][feature_id], readable)

            // display the feature name
            selection.append("text")
                .text(pretty_name)
                .attr("class", "feature-details")
                .attr("x", x + text_margin_left)
                .attr("y", features_line_y + text_offset * (i + 1))
                .attr("fill", "black")

            // display the formatted feature value
            selection.append("text")
                .text(value_text)
                .attr("class", "feature-details")
                .attr("x", x + width - text_margin_left)
                .attr("y", features_line_y + text_offset * (i + 1))
                .attr("fill", "black")
                .attr("text-anchor", "end");
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

    return my;
}