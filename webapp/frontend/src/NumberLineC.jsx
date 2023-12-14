import React, { useState } from 'react';
import Draggable from 'react-draggable';

const NumberLineC = ({ start, end, minRange, maxRange, currentValue }) => {
    // State to keep track of the current range
    const [currentRange, setCurrentRange] = useState({ min: minRange, max: maxRange });
    const [tempMinRange, setTempMinRange] = useState(minRange.toString());
    const [tempMaxRange, setTempMaxRange] = useState(maxRange.toString());
    const width = 290;
    const calc = 185;
    const height = 70;

    // Handler for dragging the Min Range Handle
    const handleMinRangeChange = (e, ui) => {
        // Calculate the new value based on the drag position
        const newMinRange = calculateValueFromPosition(ui.x); // Takes x cord. of cursor
        console.log({ newMinRange })
        setTempMinRange(newMinRange);
        // Update the state with the new value
        setCurrentRange({ ...currentRange, min: newMinRange });
    };

    // Handler for dragging the Max Range Handle
    const handleMaxRangeChange = (e, ui) => {
        // Calculate the new value based on the drag position
        const newMaxRange = calculateValueFromPosition(ui.x);  // Takes x cord. of cursor
        console.log({ newMaxRange })
        setTempMaxRange(newMaxRange);
        // Update the state with the new value
        setCurrentRange({ ...currentRange, max: newMaxRange });
    };

    // Function to calculate the value based on the drag position 
    const calculateValueFromPosition = (x) => {
        console.log({ x });
        // Calculate the percentage of the drag position relative to the window width
        const percent = (x / width) * 100; // Adjusted scaling factor for width
        // Calculate the corresponding value based on the percentage and round to the nearest integer
        return Math.round(start + (percent / 100) * (end - start));
    };

    // Function to calculate the position based on the value
    const calculatePositionFromValue = (value) => {
        // Calculate the percentage of the value relative to the start and end values
        const percent = ((value - start) / (end - start));
        // Calculate the corresponding position based on the percentage and window width
        const based = percent * width;
        return Math.round(based); // Set maximum width to 350px
    };



    // MIN: INPUT LOGIC

    // Position of minRange to for input label 
    const minRangePosition = {
        left: `${calculatePositionFromValue(currentRange.min)}px`,
        transform: 'translateX(-50%)',
        position: 'absolute',
        display: 'flex',
        top: '70%',
        //transition: 'left 0.05s ease',
    }
    // Function to handle min Range input change
    const handleMinChange = (e) => {
        setTempMinRange(e.target.value);
    };
    // Function to handle Min Range input key press
    const handleMinRangeKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleMinRangeBlur();
        }
    };
    // Function to handle Min Range input blur
    const handleMinRangeBlur = () => {
        const newValue = parseFloat(tempMinRange);
        if (!isNaN(newValue) && newValue >= start && newValue < currentRange.max && newValue !== currentRange.max) {
            setCurrentRange({ ...currentRange, min: parseInt(tempMinRange) });

        }
    };

    // MAX: INPUT LOGIC
    // Position of maxRange to for input label 
    const maxRangePosition = {
        left: `${calculatePositionFromValue(currentRange.max)}px`,
        transform: 'translateX(-50%)',
        position: 'absolute',
        display: 'flex',
        top: '70%',
        //transition: 'left 0.05s ease',
    }
    // Function to handle Max Range input change
    const handleMaxChange = (e) => {
        setTempMaxRange(e.target.value);
    };
    // Function to handle Max Range input key press
    const handleMaxRangeKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleMaxRangeBlur();
        }
    };
    // Function to handle Max Range input blur
    const handleMaxRangeBlur = () => {
        const newValue = parseFloat(tempMaxRange);
        if (!isNaN(newValue) && newValue <= end && newValue > currentRange.min && newValue !== currentRange.min) {
            setCurrentRange({ ...currentRange, max: parseInt(tempMaxRange) });
        }
    };

    return (
        <div className='number-line-container' style={{ width: `100%`, height: `100%` }}>
            {/* SVG Number Line */}
            <svg className='svg-num' width="100%" height="100%" viewBox={`-10 0 ${width+20} ${height}`}>
                {/* Number Line */}
                <line
                    x1={`${calculatePositionFromValue(start)}px`}
                    y1="50%"
                    x2={`${calculatePositionFromValue(end)}px`}
                    y2="50%"
                    style={{ stroke: '#333', strokeWidth: 2 }}
                />

                {/* Highlighted Range */}
                <rect
                    x={`${calculatePositionFromValue(currentRange.min)}px`}
                    y="45%"
                    width={`${calculatePositionFromValue(currentRange.max - currentRange.min)}px`}
                    height="10%"
                    fill="#7fc3ff"
                //style={{ transition: 'width 0.3s ease' }} 
                />

                {/* NumberLine Ticks */}
                <rect
                    x={`${calculatePositionFromValue(start)}px`} // start
                    y="40%"
                    width={`2px`}
                    height="20%"
                    fill="#333"
                />

                <rect
                    x={`${calculatePositionFromValue(end)}px`} // end 
                    y="40%"
                    width={`2px`}
                    height="20%"
                    fill="#333"
                />

                {/* Current Value Dot */}
                <circle
                    cx={`${calculatePositionFromValue(currentValue)}px`}
                    cy="50%"
                    r="5"
                    fill="black"
                />

                {/* Labels at Start and End */}
                <text
                    x={`${calculatePositionFromValue(start)}px`}
                    y="80%"
                    fontSize="12"
                    fill="#333"
                    textAnchor="middle"
                >
                    {start}
                </text>

                <text
                    x={`${calculatePositionFromValue(end)}px`}
                    y="80%"
                    fontSize="12"
                    fill="#333"
                    textAnchor="middle"
                >
                    {end}
                </text>
            </svg>
            <div className='interact-line'>
                {/* Min Range Handle */}
                <Draggable
                    axis="x"
                    bounds={{ left: calculatePositionFromValue(start), right: calculatePositionFromValue(currentRange.max) }}
                    onDrag={handleMinRangeChange}
                    position={{ x: calculatePositionFromValue(currentRange.min), y: 0 }}
            
                >
                    <div style={{ position: 'absolute', top: '43%', width: '3px', height: '16px', backgroundColor: '#333', cursor: 'ew-resize' }} />
                </Draggable>

                {/* Input for Min Range */}
                <div className='range-inputs'>
                    <div style={minRangePosition}>   {/* Sets Postion*/}
                        <input
                            type="number"
                            value={tempMinRange}
                            onChange={handleMinChange}
                            onBlur={handleMinRangeBlur}
                            onKeyDown={handleMinRangeKeyPress}
                            style={{ position: 'absolute', top: '-50px' }}
                        />
                    </div>
                </div>

                {/* Max Range Handle */}
                <Draggable
                    axis="x"
                    bounds={{ left: calculatePositionFromValue(currentRange.min), right: calculatePositionFromValue(end) }}
                    onDrag={handleMaxRangeChange}
                    position={{ x: calculatePositionFromValue(currentRange.max), y: 0 }}
              
                >
                    <div style={{ position: 'absolute', top: '43%', width: '3px', height: '16px', backgroundColor: '#333', cursor: 'ew-resize' }} />
                </Draggable>

                {/* Input for Max Range */}

                <div className='range-inputs'>
                    <div style={maxRangePosition}>   {/* Sets Postion*/}
                        <input
                            type="number"
                            value={tempMaxRange}
                            onChange={handleMaxChange}
                            onBlur={handleMaxRangeBlur}
                            onKeyDown={handleMaxRangeKeyPress}
                            style={{ position: 'absolute', transform: 'translateX(-100%)', top: '-50px' }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default NumberLineC;
