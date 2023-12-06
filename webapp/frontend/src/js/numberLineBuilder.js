import { clamp_value, create_example, feature_dists_order, pretty_value, unscale } from '../../../../visualization/src/utilities.js';
import { ExplanationTypes, OFFSET_UNSCALED, expl_colors, rect_values } from "../../../../visualization/src/values.js";
import { json, select } from "d3";
import { RangeTypes } from "./numberLineUtil.js";

const detailsURL = "http://localhost:3001/visualization/data/dataset_details.json";
const readableURL = "http://localhost:3001/visualization/data/human_readable_details.json";

export const numberLineBuilder = () => {

    const my = async (selection) => {
        const width = 800;
        const height = 375;
        const explanation = {
            "instance": {
                "x0": 0.058688930117501543,
                "x1": 0.0,
                "x2": 0.1457142857142857,
                "x3": 0.7435897435897436
            },
            "region": {
                "x0": [
                    0.0032405690290033817,
                    0.010667903814464808
                ],
                "x1": [
                    -100000000000000.0,
                    0.00019343846361152828
                ],
                "x2": [
                    0.10142857208848,
                    0.17642857134342194
                ],
                "x3": [
                    0.0769230779260397,
                    0.8717948794364929
                ]
            }
        };
        const dataset_details = await json(detailsURL);
        const readable = await json(readableURL);
        let expl_type = ExplanationTypes.Region;

        const n_features = Object.keys(dataset_details["feature_names"]).length;
        const instance = explanation["instance"];
        const region = explanation["region"];
        const [feature_distances, idx_order] = feature_dists_order(instance, region);

        const sidebar_width_ratio = 0.33;
        const sidebar_margin = 10;
        const bbox_width = width - 20;
        const bbox_height = height - 20;
        const bbox_x = 10;
        const bbox_y = 10;

        let ebox_width_ratio = 1 - sidebar_width_ratio;
        let ebox_width = (bbox_width * ebox_width_ratio) - 3 * sidebar_margin;
        let ebox_height = bbox_height - (sidebar_margin * 2);
        let ebox_x = bbox_x + (sidebar_width_ratio * bbox_width) + (2 * sidebar_margin);
        let ebox_y = bbox_y + sidebar_margin;

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

        const range_type_min = RangeTypes.Percent;
        const range_type_max = RangeTypes.Percent;
        const percent_val = 0.25;


        function get_feature_name(feature_i) {
            const feature_id = "x" + feature_i;
            const feature_name = dataset_details["feature_names"][feature_id];
            const pretty_feature_name = readable["pretty_feature_names"][feature_name];
            return pretty_feature_name
        }

        function get_line_low(value, feature_id) {
            // select the bar min value
            let min_value;
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
            let max_value;
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
            let clamped_val = value;
            clamped_val = Math.min(clamped_val, max_value);
            clamped_val = Math.max(clamped_val, min_value);
            return ((clamped_val - min_value) / (max_value - min_value)) * line_plot_width;
        }

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


            let example_val = instance_val;
            if (expl_type == ExplanationTypes.Example) {
                if (has_change) {
                    let offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
                    example_val = create_example(instance_val, region_lower, region_upper, offset);
                }
            }

            let min_plot_value;
            let max_plot_value;
            if (expl_type == ExplanationTypes.Region) {
                min_plot_value = Math.min(instance_val, region_lower);
                max_plot_value = Math.max(instance_val, region_upper);
            } else if (expl_type == ExplanationTypes.Example) {
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
                    let offset = unscale(OFFSET_UNSCALED, feature_id, dataset_details);
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
    
    return my;
}