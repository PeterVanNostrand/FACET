import React, { useEffect, useRef } from 'react';
import { select } from 'd3';
import { numberLineBuilder } from '../../js/numberLineBuilder'

const NumberLine = ({ explanation }) => {
    const numberLineRef = useRef();


    useEffect(() => {
        const selection = select(numberLineRef.current);
        const visual_display = numberLineBuilder(selection);
        selection.append('circle')
            .attr('cx', 50)
            .attr('cy', 50)
            .attr('r', 20)
            .style('fill', 'red');
        
        var svg = select('#svg_container').append("svg")
            .attr('width', 800)
            .attr('height', 375)
            .attr("fill", "white")
            .attr("id", "image_svg");

        svg.call(visual_display)

    }, []);

    return (
        <div>
            {/* <svg ref={numberLineRef} width="100" height="100"></svg> */}
            <div id="svg_container"></div>
        </div>
    );
};

export default NumberLine;
