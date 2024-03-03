import { numberLineBuilder } from '@src/js/numberLineBuilder';
import { select } from 'd3';
import { useEffect, useRef } from 'react';

const NumberLine = ({ explanation, i, id, formatDict }) => {
    const svgRef = useRef();

    useEffect(() => {
        // Check if the svg element already exists
        const existingSvg = select(svgRef.current).select('svg');

        // Use the existing svg or create a new one
        const svg = existingSvg.empty()
            ? select(svgRef.current).append('svg')
                .attr('width', 370)
                .attr('height', 60)
                .attr('fill', 'white')
                .attr('id', `image_svg_${id}`)

            : existingSvg;

        let visualDisplay = numberLineBuilder(explanation, i, formatDict);
        svg.call(visualDisplay);

        // Clean up when the component unmounts
        return () => {
            svg.selectAll('*').remove();
            svg.remove();
        };
    }, [explanation, i, id]);

    return (
        <div id={`${id}`} ref={svgRef} className="explanationLine"></div>
    );
};

export default NumberLine;
