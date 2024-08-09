// group import d3 functions
const {
    json,
    select,
} = d3;

import { save } from "./d3-save-svg.js";
import { liguisticDisplay } from "./linguisticDisplay.js";
import { numericDisplay } from "./numericDisplay.js";
import { ExplanationTypes } from "./values.js";
import { visualDisplay } from "./visualDisplay.js";

// data source url
const detailsURL = "/visualization/data/dataset_details.json";

const DisplayTypes = Object({
    Numeric: "numeric",
    Linguistic: "linguistic",
    Visual: "visual"
})

// main rendering function
const main = async () => {
    // ####################################### DATA LOADING #######################################

    // load the dataset details
    const dataset_details = await json(detailsURL);

    // load the paths to the explanations JSON.
    const explanation_paths = await json("/visualization/data/explanation_paths.json");

    // load the explanation and isolate the instance and the region. Note we use +/-100000000000000 for +/-Infinity
    var explanation_id = 0;
    var explanation = await json(explanation_paths[explanation_id]);
    var current_expl_type = ExplanationTypes.Region;

    // load human readable info for the dataset
    const readableURL = "/visualization/data/human_readable_details.json"
    const readable = await json(readableURL);

    // ####################################### RENDER EXPLANATION #######################################

    select('#instance-counter')
        .text("Instance " + explanation_id);

    const width = 800;
    const height = 375;
    const svg = select('#svg_container').append("svg")
        .attr('width', width)
        .attr('height', height)
        .attr("fill", "white")
        .attr("id", "image_svg");

    const numeric_display = numericDisplay()
        .width(width)
        .height(height)
        .explanation(explanation)
        .dataset_details(dataset_details)
        .readable(readable)
        .expl_type(current_expl_type);
    const linguistic_display = liguisticDisplay()
        .width(width)
        .height(height)
        .explanation(explanation)
        .dataset_details(dataset_details)
        .readable(readable)
        .expl_type(current_expl_type);
    const visual_display = visualDisplay()
        .width(width)
        .height(height)
        .explanation(explanation)
        .dataset_details(dataset_details)
        .readable(readable)
        .expl_type(current_expl_type);

    var current_display = DisplayTypes.Numeric;
    update_display(current_display);

    // ####################################### BUTTON CONTROLS #######################################

    function update_display(dtype) {
        if (dtype) { current_display = dtype }
        // clear the SVG and render the new display
        svg.selectAll("*").remove();
        if (current_display == DisplayTypes.Numeric) {
            svg.call(numeric_display);
        } else if (current_display == DisplayTypes.Linguistic) {
            svg.call(linguistic_display);
        }
        else if (current_display == DisplayTypes.Visual) {
            svg.call(visual_display);
        }
    }

    async function update_explantion() {
        explanation = await json(explanation_paths[explanation_id]);
        // update all the displays
        numeric_display.explanation(explanation)
        linguistic_display.explanation(explanation)
        visual_display.explanation(explanation)
        // clear the SVG and render the correct display
        update_display();
    }

    function update_expl_type(etype) {
        current_expl_type = etype;
        // update all the displays
        numeric_display.expl_type(etype)
        linguistic_display.expl_type(etype)
        visual_display.expl_type(etype)
        // clear the SVG and render the correct display
        update_display();
    }

    // numeric button
    d3.select('button#numeric').on('click', function () {
        update_display(DisplayTypes.Numeric);
    });

    // linguistic button
    d3.select('button#linguistic').on('click', function () {
        update_display(DisplayTypes.Linguistic);
    });

    // visual button
    d3.select('button#visual').on('click', function () {
        update_display(DisplayTypes.Visual);
    });

    // example button
    d3.select('button#example').on('click', function () {
        update_expl_type(ExplanationTypes.Example);
    });

    // region button
    d3.select('button#region').on('click', function () {
        update_expl_type(ExplanationTypes.Region);
    });

    function execute_save() {
        var config = {
            filename: 'explanation_' + String(explanation_id).padStart(3, 0) + "_" + current_display + "_" + current_expl_type,
        }
        save(d3.select('svg').node(), config);
    }

    // save button
    d3.select('button#export').on('click', function () {
        execute_save();
    });

    // previous button
    d3.select('button#previous').on('click', async function () {
        if (explanation_id > 0) {
            explanation_id -= 1
            select('#instance-counter')
                .text("Instance " + explanation_id);
            update_explantion();
        }
    });

    // next button
    d3.select('button#next').on('click', function () {
        if (explanation_id < explanation_paths.length - 1) {
            explanation_id += 1
            select('#instance-counter')
                .text("Instance " + explanation_id);
            update_explantion();
        }
    });

    // save all button
    d3.select('button#save-all').on('click', function () {
        // store the current display + type
        const prev_style = current_display;
        const prev_type = current_expl_type;

        // save each style and display to an SVG 
        const viz_styles = [DisplayTypes.Numeric, DisplayTypes.Linguistic, DisplayTypes.Visual];
        const expl_types = [ExplanationTypes.Example, ExplanationTypes.Region];

        for (let i = 0; i < viz_styles.length; i++) {
            const style = viz_styles[i];
            current_display = style;
            update_display(style);
            for (let j = 0; j < expl_types.length; j++) {
                const type = expl_types[j];
                current_expl_type = type;
                update_expl_type(type);
                execute_save();
            }
        }

        // restore the display + type
        current_display = prev_style;
        current_expl_type = prev_type;
    });

}

main();