import React, { useEffect, useRef } from 'react';
import { visualDisplay } from '../../../visualization/src/visualDisplay.js';
import * as d3 from 'd3'

import readable from "../../../visualization/data/human_readable_details.json";
import datasetDetails from "../../../visualization/data/dataset_details.json";


const NumberLine = ({ explanationData }) => {
    const svgRef = useRef();

    useEffect(() => {
        const explType = 'region'

        if (explanationData && datasetDetails && readable) {
            const width = 800;
            const height = 375;

            const visualization = visualDisplay()
                .width(width)
                .height(height)
                .explanation(explanationData)
                .dataset_details(datasetDetails)
                .readable(readable)
                .expl_type(explType);

            const svg = d3.select(svgRef.current)
                .attr('width', width)
                .attr('height', height);

            svg.call(visualization);
        }
    }, [explanationData]);

    return <svg ref={svgRef} />;
};

export default NumberLine;
