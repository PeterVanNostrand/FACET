import { TableTypes, explanationTable } from "./explanationTable.js";
import { sideBar } from "./sideBar.js";
import { feature_dists_order } from "./utilities.js";
import { rect_values } from "./values.js";

export const numericDisplay = () => {
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
            .attr("filter", "url(#dropshadow)");

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


        // ####################################### EXPLANATION CONTENT #######################################

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

        // compute the unscaled distance along each dimension
        const [feature_distances, idx_order] = feature_dists_order(instance, region);

        // table shape values
        const tr_width = 200
        const tr_height = 70
        const table_margin = (ebox_width - (2 * tr_width)) / 3

        // APPLICATION table
        const text_margin_top = 40;
        const atable_x_offset = table_margin
        const atable_y = ebox_y + text_margin_top
        const atable_x = ebox_x + atable_x_offset

        const atable = explanationTable()
            .width(tr_width)
            .height(tr_height)
            .x(atable_x)
            .y(atable_y)
            .explanation(explanation)
            .dataset_details(dataset_details)
            .readable(readable)
            .expl_type(expl_type)
            .idx_order(idx_order)
            .feature_distances(feature_distances)
            .table_type(TableTypes.Application);
        selection.call(atable);

        // EXPLANATION table
        const etable_x_offset = (table_margin * 2) + tr_width
        const etable_y = ebox_y + text_margin_top
        const etable_x = ebox_x + etable_x_offset
        const etable = explanationTable()
            .width(tr_width)
            .height(tr_height)
            .x(etable_x)
            .y(etable_y)
            .explanation(explanation)
            .dataset_details(dataset_details)
            .readable(readable)
            .expl_type(expl_type)
            .idx_order(idx_order)
            .feature_distances(feature_distances)
            .table_type(TableTypes.Explanation);
        selection.call(etable);
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