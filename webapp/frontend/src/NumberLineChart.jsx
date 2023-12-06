import React, { useState, useEffect } from 'react';
import { VictoryChart, VictoryAxis, VictoryLine, VictoryScatter, VictoryTheme, VictoryLabel} from 'victory';

const NumberLineChart = ({ start, end, initialMinRange, initialMaxRange, currentValue }) => {
  // minRange, maxRange to be used for backEnd
  const [minRange, setMinRange] = useState(initialMinRange);
  const [maxRange, setMaxRange] = useState(initialMaxRange);
  // temp. minRange and maxRange used for inputs
  const [tempMinRange, setTempMinRange] = useState(initialMinRange.toString());
  const [tempMaxRange, setTempMaxRange] = useState(initialMaxRange.toString());

  useEffect(() => {
    setTempMinRange(minRange.toString());
    setTempMaxRange(maxRange.toString());
  }, [minRange, maxRange]);

  // Handle Changing Min and Max 
  const handleMinRangeChange = (event) => {
    setTempMinRange(event.target.value);
  };
  const handleMaxRangeChange = (event) => {
    setTempMaxRange(event.target.value);
  };
  // BLUR:
  const handleMinRangeBlur = () => {
    const newMinRange = parseFloat(tempMinRange);
    // Tests if its not empty, that the min is not less than the start-val or greater than maxRange 
    if (!isNaN(newMinRange) && newMinRange >= start && newMinRange < maxRange) {
      setMinRange(newMinRange);
  }
};
  const handleMaxRangeBlur = () => {
    const newMaxRange = parseFloat(tempMaxRange);
    if (!isNaN(newMaxRange) && newMaxRange <= end && newMaxRange > minRange && newMaxRange !== minRange) {
      setMaxRange(newMaxRange);
    }
  };
  // KEY PRESS: 
  const handleMaxRangeKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleMaxRangeBlur();
    }
  };
  const handleMinRangeKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleMinRangeBlur();
    }
  };
  // POSITION:
  // Calculating Position of minRange/maxRange on numberline (based on %)
  const minRangePosition = {
    left: `${((minRange - start) / (end - start)) * 100}%`,
    transform: 'translateX(-50%)',
    position: 'absolute',
    top: '10px',
    display: 'flex',
    transition: 'left 0.5s ease',
  };
  const maxRangePosition = {
    left: `${((maxRange - start) / (end - start)) * 100}%`,
    transform: 'translateX(-50%)',
    position: 'absolute',
    top: '10px',
    display: 'flex',
    transition: 'left 0.5s ease', 
  };

  return (
    <div className='vc-chart' style={{ position: 'relative' }}>
      <VictoryChart theme={VictoryTheme.material} height={100}>
        {/* Set up Chart*/}
        <VictoryAxis
          tickValues={[start, end]}
          tickFormat={(tick) => tick.toFixed(2)}
          tickLength={0}
          style={{
            axis: { stroke: 'none' },
            ticks: { stroke: 'none' },
            tickLabels: { fontSize: 12, padding: 5 },
          }}
        />
           {/* Rendering Number Line */}
        <VictoryLine classname="number-line"
          data={[{ x: start, y: 0 }, { x: end, y: 0 }]}
          style={{ data: { stroke: 'grey', strokeWidth: 2 } }}
        />
              {/* Rendering Selected Range */}
           <VictoryLine className="selected-range"
          data={[{ x: minRange, y: 0 }, { x: maxRange, y: 0 }]}
          style={{ data: { stroke: 'blue', strokeWidth: 2 } }}
          animate={{ duration: 400, easing: 'linear' }}
        />

        {/* Current Dot */}
        <VictoryScatter data={[{ x: parseFloat(currentValue), y: 0 }]} size={3} style={{ data: { fill: 'black' } 
          }}/>
        <VictoryScatter
          data={[{ x: parseFloat(currentValue), y: -1 }]}
          size={3}
          style={{ labels: { fill: "black", fontSize: 12 } }}
          labels={currentValue} 
          labelComponent={<VictoryLabel dy={11} />}
        />

      </VictoryChart>
        {/* Inputs for Range */}
      <div className='range-inputs'>
        <div style={minRangePosition}>     {/* Sets Postion*/}
          <label></label>
          <input
            type="number"
            value={tempMinRange}
            onChange={handleMinRangeChange}
            onKeyDown={handleMinRangeKeyPress}
            onBlur={handleMinRangeBlur}
          />
        </div>

        <div style={maxRangePosition}>   {/* Sets Postion*/}
          <label></label>
          <input
            type="number"
            value={tempMaxRange}
            onChange={handleMaxRangeChange}
            onKeyDown={handleMaxRangeKeyPress}
            onBlur={handleMaxRangeBlur}
          />
        </div>
      </div>
    </div>
  );
};

export default NumberLineChart;
