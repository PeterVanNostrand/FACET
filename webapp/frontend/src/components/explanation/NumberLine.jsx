import React, { useEffect } from 'react';
import { select } from 'd3';
import { numberLineBuilder } from '../../js/numberLineBuilder';

const NumberLine = ({ explanation, i, id }) => {
    console.log(explanation)
        
    useEffect(() => {
        // Check if the svg element already exists
        const svgContainer = select(`#${id}`);
        const existingSvg = svgContainer.select('svg');

        // Use the existing svg or create a new one
        const svg = existingSvg.empty()
            ? svgContainer.append('svg')
                .attr('width', 400)
                .attr('height', 60)
                .attr('fill', 'white')
                .attr('id', 'image_svg')
            : existingSvg;

        console.log();
        let visualDisplay = numberLineBuilder(explanation, i);
        svg.call(visualDisplay);

        // Clean up when the component unmounts
        return () => {
            svg.selectAll('*').remove();
        };
    }, [explanation]);

    return (
        <div id="svg_container"></div>
    );
};

export default NumberLine;
