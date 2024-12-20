import { RangeTypes } from "@src/js/numberLineUtil.js";
import { ExplanationTypes, clamp_value, feature_dists_order, formatValue } from "@src/js/utilities.js";

export const rect_values = {
    fill: "white",
    round: 5,
    stroke_width: 2,
    stroke: "black",
}

export const expl_colors = {
    desired: "#006eff",
    undesired: "#e73b3c",
    altered_bad: "#e73b3c",
    altered_good: "#006eff",
    unaltered: "#939393"
}

export const numberLineBuilder = (explanation, index, formatDict) => {

    const my = async (selection) => {

        const width = 50;
        let expl_type = ExplanationTypes.Region;

        const instance = explanation["instance"];
        const region = explanation["region"];
        const [feature_distances, idx_order] = feature_dists_order(instance, region);

        const sidebar_width_ratio = 0.33;
        const sidebar_margin = 10;
        const bbox_width = width - 20;
        const bbox_x = 10;
        const bbox_y = 10;

        let ebox_x = bbox_x + (sidebar_width_ratio * bbox_width) + (2 * sidebar_margin);
        let ebox_y = bbox_y + sidebar_margin;

        const labels_x = ebox_x + 0;
        const labels_width = 0
        const line_plot_pad_x = 0;
        const line_plot_pad_y = 10;
        const line_plot_x = labels_x + labels_width + line_plot_pad_x;
        const line_plot_width = 290;
        const line_spacing = 55;
        const line_width = 2.4;
        const tick_height = 16;
        const bar_height = tick_height - 3;
        const circle_radius = tick_height / 2 - 1;
        const value_font = 14;
        const font_family = "Roboto"

        const range_type_min = RangeTypes.Percent;
        const range_type_max = RangeTypes.Percent;
        const percent_val = 0.25;


        function get_line_low(value, feature_id) {
            // select the bar min value
            let min_value;
            if (range_type_min == RangeTypes.DataSet) {
                min_value = formatDict["min_values"][feature_id];
            } else if (range_type_min == RangeTypes.StdDev) {
                min_value = value - formatDict["std_dev"][feature_id];
            } else if (range_type_min == RangeTypes.Percent) {
                min_value = value - (percent_val * value);
            }
            min_value = clamp_value(min_value, feature_id, formatDict, formatDict);
            return min_value;
        }

        function get_line_high(value, feature_id) {
            // select the bar max value
            let max_value;
            if (range_type_max == RangeTypes.DataSet) {
                max_value = formatDict["max_values"][feature_id];
            } else if (range_type_max == RangeTypes.StdDev) {
                max_value = value + formatDict["std_dev"][feature_id];
            }
            else if (range_type_max == RangeTypes.Percent) {
                max_value = value + (percent_val * value);
                if (value == 0 || value == 0.01 || value == -0.01) { // just the lock offset
                    max_value = value + formatDict["std_dev"][feature_id];
                }
            }
            max_value = clamp_value(max_value, feature_id, formatDict, formatDict);
            return max_value;
        }

        function pixel_scale(value, min_value, max_value) {
            let clamped_val = value;
            clamped_val = Math.min(clamped_val, max_value);
            clamped_val = Math.max(clamped_val, min_value);
            return ((clamped_val - min_value) / (max_value - min_value)) * line_plot_width;
        }

        const feature_id = "x" + idx_order[index];
        const feature_name = formatDict["feature_names"][feature_id];


        // ##### DRAW THE NUMBER LINE ########################################################################

        // draw a number line for this feature
        const line_y = ebox_y + line_plot_pad_y + line_spacing * 0;
        selection.append("line")
            .attr("x1", line_plot_x)
            .attr("x2", line_plot_x + line_plot_width)
            .attr("y1", line_y)
            .attr("y2", line_y)
            .attr("stroke", rect_values.stroke)
            .attr("stroke-width", line_width);
        //left and right ticks
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
        const instance_val = instance[feature_id]
        const region_lower = region[feature_id][0]
        const region_upper = region[feature_id][1]

        const has_change = feature_distances[idx_order[index]] > 0;


        let min_plot_value;
        let max_plot_value;
        if (expl_type == ExplanationTypes.Region) {
            min_plot_value = Math.min(instance_val, region_lower);
            max_plot_value = Math.max(instance_val, region_upper);
        }

        const line_min = get_line_low(min_plot_value, feature_id);
        const line_max = get_line_high(max_plot_value, feature_id);

        const bar_lower_val = Math.max(region_lower, line_min);
        const bar_upper_val = Math.min(region_upper, line_max);

        // TEXT LABEL the ends of the explanation bar


        const line_text_lower = selection.append("text")
            .text(formatValue(line_min, feature_id, formatDict))
            .attr("class", "svgtext")
            .attr("font-size", value_font)
            .attr("font-family", font_family) // Use Inter font family
            .attr("fill", "black")
            .attr("x", line_plot_x)
            .attr("y", line_y + bar_height + value_font)
            .attr("text-anchor", "middle")
            .attr("class", "tick-label");
        if (line_min == region_lower || line_min == instance_val) {
            line_text_lower.text("");
        }

        const line_text_upper = selection.append("text")
            .text(formatValue(line_max, feature_id, formatDict))
            .attr("font-size", value_font)
            .attr("font-family", font_family) // Use Inter font family
            .attr("fill", "black")
            .attr("x", line_plot_x + line_plot_width)
            .attr("y", line_y + bar_height + value_font)
            .attr("text-anchor", "middle")
            .attr("class", "tick-label");
        if (line_max == region_upper || line_max == instance_val) {
            line_text_upper.text("");
        }

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

            // TEXT LABELS for min/max of the region explanation
            const bar_text_lower = selection.append("text")
                .text(formatValue(bar_lower_val, feature_id, formatDict))
                .attr("font-size", value_font)
                .attr("fill", bar_color)
                .attr("x", bar_start_px)
                .attr("y", line_y - bar_height)
                .attr("text-anchor", "end")
                .attr("class", "tick-label");

            const bar_text_upper = selection.append("text")
                .text(formatValue(bar_upper_val, feature_id, formatDict))
                .attr("font-size", value_font)
                .attr("fill", bar_color)
                .attr("x", bar_end_px)
                .attr("y", line_y - bar_height)
                .attr("text-anchor", "start")
                .attr("class", "tick-label");

            if ((bar_upper_val - bar_lower_val) < Math.abs(0.001 * bar_upper_val)) {
                // Hide the upper text if values are equal
                bar_text_upper.remove();
                bar_text_lower.attr("text-anchor", "middle");
            }

            // if we have space for the text, center the bar end labels on the ticks
            const lower_text_width = bar_text_lower.node().getComputedTextLength();
            const upper_text_width = bar_text_upper.node().getComputedTextLength();
            if (bar_width_px > ((lower_text_width / 2) + (upper_text_width / 2) + 10)) {
                bar_text_lower.attr("text-anchor", "middle");
                bar_text_upper.attr("text-anchor", "middle");
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
            .text(formatValue(instance_val, feature_id, formatDict))
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
    return my;
}