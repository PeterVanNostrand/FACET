import React, { useEffect, useRef } from 'react';
import { visualDisplay } from '../../../visualization/src/visualDisplay.js';
import * as d3 from 'd3'
import { json } from 'd3';

const ExplanationComponent = ({ explanationData, datasetDetails, explType }) => {
    const svgRef = useRef();

    useEffect(() => {
        const readableURL = "/visualization/data/human_readable_details.json";
        const readable = async () => await json(readableURL);

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
    }, [explanationData, datasetDetails, readable, explType]);

    return <svg ref={svgRef} />;
};

export default ExplanationComponent;
