import React, { useState, useEffect } from 'react';
import { VictoryChart, VictoryAxis, VictoryLine, VictoryScatter, VictoryTheme, VictoryLabel} from 'victory';
import './style.css';

const NumberLineChart = ({ start, end, initialMinRange, initialMaxRange, currentValue }) => {
  // Set up useState
  const [minRange, setMinRange] = useState(initialMinRange);
  const [maxRange, setMaxRange] = useState(initialMaxRange);
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

  const handleMinRangeBlur = () => {
    const newMinRange = parseFloat(tempMinRange);
    if (!isNaN(newMinRange) && newMinRange >= start && newMinRange < maxRange) {
      setMinRange(newMinRange);
    }
    else {  
    }
  };

  const handleMaxRangeChange = (event) => {
    setTempMaxRange(event.target.value);
  };

  const handleMaxRangeBlur = () => {
    const newMaxRange = parseFloat(tempMaxRange);
    if (!isNaN(newMaxRange) && newMaxRange <= end && newMaxRange > minRange) {
      setMaxRange(newMaxRange);
    }
    if (newMaxRange == minRange)  {
      
    }
  };

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
        {/* Rendering Number Line */}
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
   
        <VictoryLine classname="number-line"
          data={[{ x: start, y: 0 }, { x: end, y: 0 }]}
          style={{ data: { stroke: 'grey', strokeWidth: 2 } }}
        />
      
           <VictoryLine className="selected-range"
          data={[{ x: minRange, y: 0 }, { x: maxRange, y: 0 }]}
          style={{ data: { stroke: 'blue', strokeWidth: 2 } }}
          animate={{ duration: 400, easing: 'linear' }}
        />

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

      <div className='range-inputs'>
        <div style={minRangePosition}>
          <label></label>
          <input
            type="number"
            value={tempMinRange}
            onChange={handleMinRangeChange}
            onKeyDown={handleMinRangeKeyPress}
            onBlur={handleMinRangeBlur}
          />
        </div>

        <div style={maxRangePosition}>
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
