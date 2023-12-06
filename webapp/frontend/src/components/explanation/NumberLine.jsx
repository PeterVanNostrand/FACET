import React, { useEffect } from 'react';
import { select } from 'd3';
import { numberLineBuilder } from '../../js/numberLineBuilder';

const NumberLine = ({ explanation }) => {
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

    const visualDisplay = numberLineBuilder(svg);
    svg.call(visualDisplay);

    // Clean up when the component unmounts
    return () => {
      svg.selectAll('*').remove();
    };
  }, [explanation]);

  return (
    <div>
      <div id="svg_container"></div>
    </div>
  );
};

export default NumberLine;
