import React, { useEffect } from 'react';
import { select } from 'd3';
import { numberLineBuilder } from '../../js/numberLineBuilder';

const NumberLine = ({ explanation, i }) => {
    useEffect(() => {
        // Check if the svg element already exists
        const svgContainer = select('#svg_container');
        const existingSvg = svgContainer.select('svg');

        // Use the existing svg or create a new one
        const svg = existingSvg.empty()
            ? svgContainer.append('svg')
                .attr('width', 800)
                .attr('height', 375)
                .attr('fill', 'white')
                .attr('id', 'image_svg')
            : existingSvg;


        const explanation = {
            "instance": {
                "x0": 4895,
                "x1": 0.0,
                "x2": 10200,
                "x3": 360
            },
            "region": {
                "x0": [
                    412, 1013
                ],
                "x1": [
                    0, 8
                ],
                "x2": [
                    7100, 12350
                ],
                "x3": [
                    48, 420
                ]
            }
        };

        const explanation0 = {
            "instance": 4895,
            "region": [412, 1013]
        }
    
        const explanation1 = {
            "instance": 0.0,
            "region": [0, 8]
        }
    
        const explanation2 = {
            "instance": 10200,
            "region": [7100, 12350]
        }
    
        const explanation3 = {
            "instance": 360,
            "region": [48, 420]
        }

        const visualDisplay = numberLineBuilder(explanation, i);
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
