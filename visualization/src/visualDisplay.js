import { sideBar } from "./sideBar.js";
import { clamp_value, create_example, feature_dists_order, pretty_value, unscale } from "./utilities.js";
import { ExplanationTypes, OFFSET_UNSCALED, expl_colors, rect_values } from "./values.js";

export const visualDisplay = () => {
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

        const RangeTypes = {
            DataSet: "dataset", // use min/max value from dataset
            StdDev: "stddev", // use +/- 1 s.d. from the value
            Percent: "percent", // use a percentage increase
        };
        const range_type_min = RangeTypes.Percent;
        const range_type_max = RangeTypes.Percent;
        const percent_val = 0.25;

        function get_line_low(value, feature_id) {
            // select the bar min value
            var min_value;
            if (range_type_min == RangeTypes.DataSet) {
                min_value = dataset_details["min_values"][feature_id];
            } else if (range_type_min == RangeTypes.StdDev) {
                min_value = value - dataset_details["std_dev"][feature_id];
            } else if (range_type_min == RangeTypes.Percent) {
                min_value = value - (percent_val * value);
            }
            min_value = clamp_value(min_value, feature_id, readable, dataset_details);
            return min_value;
        }

        function get_line_high(value, feature_id) {
            // select the bar max value
            var max_value;
            if (range_type_max == RangeTypes.DataSet) {
                max_value = dataset_details["max_values"][feature_id];
            } else if (range_type_max == RangeTypes.StdDev) {
                max_value = value + dataset_details["std_dev"][feature_id];
            }
            else if (range_type_max == RangeTypes.Percent) {
                max_value = value + (percent_val * value);
            }
            max_value = clamp_value(max_value, feature_id, readable, dataset_details);
            return max_value;
        }

        function pixel_scale(value, min_value, max_value) {
            var clamped_val = value;
            clamped_val = Math.min(clamped_val, max_value);
            clamped_val = Math.max(clamped_val, min_value);
            return ((clamped_val - min_value) / (max_value - min_value)) * line_plot_width;
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

        // add header bar
        const header_line_y = ebox_y + 50
        const ebox_padding = 15
        selection.append("line")
            .attr("x1", ebox_x + ebox_padding)
            .attr("x2", ebox_x + ebox_width - ebox_padding)
            .attr("y1", header_line_y)
            .attr("y2", header_line_y)
            .attr("stroke", rect_values.stroke)
            .attr("stroke-width", 2);

        // Current status (rejected)
        var bad_header = selection.append("text")
            .attr("x", ebox_x + ebox_padding + 5)
            .attr("y", header_line_y - 10)
            .attr("class", "feature-details")
            .attr("class", "table-header")
            .attr("font-size", 16)
            .attr("font-style", "bold")
            .attr("fill", "black")
        bad_header.append("tspan")
            .text(readable["scenario_terms"]["instance_name"] + " (")
        bad_header.append("tspan")
            .text(readable["scenario_terms"]["undesired_outcome"])
            .attr("fill", expl_colors.undesired);
        bad_header.append("tspan")
            .text(")")
            .attr("fill", "black");

        // explanation status (accepted)
        var good_header = selection.append("text")
            .attr("x", ebox_x + ebox_width - ebox_padding - 5)
            .attr("y", header_line_y - 10)
            .attr("class", "feature-details")
            .attr("class", "table-header")
            .attr("font-size", 16)
            .attr("font-style", "bold")
            .attr("fill", "black")
            .attr("text-anchor", "end")
        good_header.append("tspan")
            .text("Explanation (")
        good_header.append("tspan")
            .text(readable["scenario_terms"]["desired_outcome"])
            .attr("fill", expl_colors.desired);
        good_header.append("tspan")
            .text(")")
            .attr("fill", "black");

        // compute the unscaled distance along each dimension
        const [feature_distances, idx_order] = feature_dists_order(instance, region);

        const labels_x = ebox_x + 10;
        const labels_width = 150;
        const line_plot_pad_x = 10;
        const line_plot_pad_y = 100;
        const line_plot_x = labels_x + labels_width + line_plot_pad_x;
        const line_plot_width = 290;
        const line_spacing = 55;
        const line_width = 1;
        const tick_height = 10;
        const bar_height = tick_height - 2;
        const circle_radius = bar_height / 2;
        const value_font = 12;

        for (let i = 0; i < n_features; i++) { // for each feature
            const feature_id = "x" + idx_order[i];
            const feature_name = dataset_details["feature_names"][feature_id];

            // ##### DRAW THE NUMBER LINE #####

            // draw a number line for this feature
            const line_y = ebox_y + line_plot_pad_y + line_spacing * i;
            selection.append("line")
                .attr("x1", line_plot_x)
                .attr("x2", line_plot_x + line_plot_width)
                .attr("y1", line_y)
                .attr("y2", line_y)
                .attr("stroke", rect_values.stroke)
                .attr("stroke-width", line_width);

            // add a text label for the line
            selection.append("text")
                .text(get_feature_name(idx_order[i]) + ":")
                .attr("x", labels_x + labels_width)
                .attr("y", line_y + (tick_height / 2))
                .attr("class", "feature-details")
                .attr("fill", "black")
                .attr("text-anchor", "end");
            // add ticks to the ends of the line and label them
            selection.append("line")
                .attr("x1", line_plot_x)
                .attr("x2", line_plot_x)
                .attr("y1", line_y - tick_height / 2)
                .attr("y2", line_y + tick_height / 2)
                .attr("stroke", rect_values.stroke)
                .attr("stroke-width", line_width);
            selection.append("line")
                .attr("x1", line_plot_x + line_plot_width)
                .attr("x2", line_plot_x + line_plot_width)
                .attr("y1", line_y - tick_height / 2)
                .attr("y2", line_y + tick_height / 2)
                .attr("stroke", rect_values.stroke)
                .attr("stroke-width", line_width);


            // determine how to scale the numberline
            const instance_val = unscale(instance[feature_id], feature_id, dataset_details);
            const region_lower = unscale(region[feature_id][0], feature_id, dataset_details);
            const region_upper = unscale(region[feature_id][1], feature_id, dataset_details);

            const has_change = feature_distances[idx_order[i]] > 0;


            var example_val = instance_val;
            if (expl_type == ExplanationTypes.Example) {
                if (has_change) {
                    var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                    example_val = create_example(instance_val, region_lower, region_upper, offset);
                }
            }

            var min_plot_value;
            var max_plot_value;
            if (expl_type == ExplanationTypes.Region) {
                min_plot_value = Math.min(instance_val, region_lower);
                max_plot_value = Math.max(instance_val, region_upper);
            }
            else if (expl_type == ExplanationTypes.Example) {
                min_plot_value = Math.min(instance_val, example_val);
                max_plot_value = Math.max(instance_val, example_val);
                if (max_plot_value == 0) {
                    max_plot_value = 0.5 * region_upper;
                }
            }

            const line_min = get_line_low(min_plot_value, feature_id);
            const line_max = get_line_high(max_plot_value, feature_id);

            const bar_lower_val = Math.max(region_lower, line_min);
            const bar_upper_val = Math.min(region_upper, line_max);

            // label the ends of the line
            const line_text_lower = selection.append("text")
                .text(pretty_value(line_min, feature_name, readable))
                .attr("font-size", value_font)
                .attr("fill", "black")
                .attr("x", line_plot_x)
                .attr("y", line_y + bar_height + value_font)
                .attr("text-anchor", "middle")
                .attr("class", "tick-label");
            const line_text_upper = selection.append("text")
                .text(pretty_value(line_max, feature_name, readable))
                .attr("font-size", value_font)
                .attr("fill", "black")
                .attr("x", line_plot_x + line_plot_width)
                .attr("y", line_y + bar_height + value_font)
                .attr("text-anchor", "middle")
                .attr("class", "tick-label");

            // ########## EXPLANTION CONTENT ##########



            // ##### DRAW THE BAR #####
            if (expl_type == ExplanationTypes.Region) {
                // add a bar for the region range
                const lower_px = pixel_scale(region_lower, line_min, line_max)
                const upper_px = pixel_scale(region_upper, line_min, line_max)
                const bar_color = has_change ? expl_colors.altered_good : expl_colors.unaltered;
                const bar_width_px = upper_px - lower_px
                const bar_start_px = line_plot_x + lower_px;
                const bar_end_px = bar_start_px + bar_width_px;
                selection.append("rect")
                    .attr("x", bar_start_px)
                    .attr("y", line_y - (bar_height / 2))
                    .attr("width", bar_width_px, feature_id)
                    .attr("height", bar_height)
                    .attr("fill", bar_color);

                // add ticks to the ends of the bar
                selection.append("line")
                    .attr("x1", bar_start_px)
                    .attr("x2", bar_start_px)
                    .attr("y1", line_y - tick_height / 2)
                    .attr("y2", line_y + tick_height / 2)
                    .attr("stroke", rect_values.stroke)
                    .attr("stroke-width", line_width);
                selection.append("line")
                    .attr("x1", bar_end_px)
                    .attr("x2", bar_end_px)
                    .attr("y1", line_y - tick_height / 2)
                    .attr("y2", line_y + tick_height / 2)
                    .attr("stroke", rect_values.stroke)
                    .attr("stroke-width", line_width);

                // label the ends of the bar
                const bar_text_lower = selection.append("text")
                    .text(pretty_value(bar_lower_val, feature_name, readable))
                    .attr("font-size", value_font)
                    .attr("fill", bar_color)
                    .attr("x", bar_start_px)
                    .attr("y", line_y - bar_height)
                    .attr("text-anchor", "end")
                    .attr("class", "tick-label");
                const bar_text_upper = selection.append("text")
                    .text(pretty_value(bar_upper_val, feature_name, readable))
                    .attr("font-size", value_font)
                    .attr("fill", bar_color)
                    .attr("x", bar_end_px)
                    .attr("y", line_y - bar_height)
                    .attr("text-anchor", "start")
                    .attr("class", "tick-label");

                // if we have space for the text, center the bar end labels on the ticks
                const lower_text_width = bar_text_lower.node().getComputedTextLength();
                const upper_text_width = bar_text_upper.node().getComputedTextLength();
                if (bar_width_px > ((lower_text_width / 2) + (upper_text_width / 2) + 10)) {
                    bar_text_lower.attr("text-anchor", "middle");
                    bar_text_upper.attr("text-anchor", "middle");
                }
            }
            // ##### OR DRAW THE EXAMPLE CIRCLE #####
            else if (expl_type == ExplanationTypes.Example) {
                if (has_change) {
                    var offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                    // const example_val = create_example(instance_val, bar_lower_val, bar_upper_val, offset);
                    const expl_circle_x = line_plot_x + pixel_scale(example_val, line_min, line_max);
                    // draw the circle
                    selection.append("circle")
                        .attr("cx", expl_circle_x)
                        .attr("cy", line_y)
                        .attr("r", circle_radius)
                        .attr("fill", expl_colors.altered_good);
                    // add a text label
                    selection.append("text")
                        .text(pretty_value(example_val, feature_name, readable))
                        .attr("font-size", value_font)
                        .attr("fill", expl_colors.altered_good)
                        .attr("x", expl_circle_x)
                        .attr("y", line_y - bar_height)
                        .attr("text-anchor", "middle")
                        .attr("class", "tick-label");
                }
            }

            // ##### DRAW THE INSTANCE CIRCLE #####

            // add a circle for the instance value
            const unaltered_color = expl_type == ExplanationTypes.Region ? "black" : expl_colors.unaltered;
            const circle_color = has_change ? expl_colors.altered_bad : unaltered_color;
            const circle_x = line_plot_x + pixel_scale(instance_val, line_min, line_max)
            selection.append("circle")
                .attr("cx", circle_x)
                .attr("cy", line_y)
                .attr("r", circle_radius)
                .attr("fill", circle_color);

            // add text label for instance circle
            const circle_text = selection.append("text")
                .text(pretty_value(instance_val, feature_name, readable))
                .attr("font-size", value_font)
                .attr("fill", circle_color)
                .attr("x", circle_x)
                .attr("y", line_y + bar_height + value_font)
                .attr("text-anchor", "middle")
                .attr("class", "tick-label");
            // check if the instance's text label will overlap the text label from either end of the line
            if (instance_val != line_min && instance_val != line_max) {
                // get the size of the rendered text elements
                const line_lower_text_px = line_text_lower.node().getComputedTextLength() / 2;
                const line_upper_text_px = line_text_lower.node().getComputedTextLength() / 2;
                const circle_text_px = circle_text.node().getComputedTextLength() / 2;
                // compute the start/end of the upper/lower line text labels
                const line_text_up_start = line_plot_x + line_plot_width - line_upper_text_px;
                const line_text_low_end = line_plot_x + line_lower_text_px;
                // compute the size of the overlap
                const text_gap_up = line_text_up_start - (circle_x + circle_text_px)
                const text_gap_low = (circle_x - circle_text_px) - line_text_low_end
                // if there are fewer than 10 pixels between the text elements, adjust the text position
                const buffer_size = 10;
                if (text_gap_up < buffer_size) {
                    const adjusted_x = line_text_up_start - circle_text_px - buffer_size
                    circle_text.attr("x", adjusted_x)
                }
                if (text_gap_low < buffer_size) {
                    const adjusted_x = line_text_low_end + circle_text_px + buffer_size
                    circle_text.attr("x", adjusted_x)
                }
            }
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

    return my;
}