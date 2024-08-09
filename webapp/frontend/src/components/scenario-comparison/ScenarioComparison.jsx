import Autocomplete from '@mui/material/Autocomplete';
import { useState, useEffect } from 'react';
import TextField from '@mui/material/TextField';
import { StyledExplanationSlider } from './StyledComponents.jsx';
import './scenariocomparison.css';

const ScenarioComparison = ({ savedScenarios }) => {
    const [scenarioOne, setScenarioOne] = useState([]);
    const [scenarioTwo, setScenarioTwo] = useState([]);
    const [scenarioOneIndex, setScenarioOneIndex] = useState(scenarioOne.explanationIndex);
    const [scenarioTwoIndex, setScenarioTwoIndex] = useState(scenarioTwo.explanationIndex);

    // Handle Autocomplete selection change
    const handleDropDownOne = (event, newScenario) => {
        if (newScenario) {
            const scenarioIndex = parseInt(newScenario.split(' ')[1]);
            const selectedScenario = savedScenarios[scenarioIndex];
            setScenarioOne(selectedScenario);

        }
    };

    const handleDropDownTwo = (event, newScenario) => {
        if (newScenario) {
            const scenarioIndex = parseInt(newScenario.split(' ')[1]);
            const selectedScenario = savedScenarios[scenarioIndex];
            setScenarioTwo(selectedScenario);

        }
    };

    const handleDropDownIndexOne = (event, newIndex) => {
        if (newIndex) {
            setScenarioOneIndex(newIndex);
        }
    };

    const handleDropDownIndexTwo = (event, newIndex) => {
        if (newIndex !== undefined) {
            setScenarioTwoIndex(newIndex);
        }
    };
    // SLIDER: 
    const rangeText = (range) => {
        return `$${range}`;
    }

    const createMarks = (features, xid, units, values) => {
        const feature = features.find(feature => feature.xid === xid);

        const slider_marks = [
            {
                value: Math.round(feature.min),
                label: `${units === '$' ? `${units}${Math.round(feature.min)}` : `${Math.round(feature.min)} ${units}`}`,
            },
            {
                value: Math.round(feature.current_value),
                label: `${units === '$' ? `${units}${Math.round(feature.current_value)}` : `${Math.round(feature.current_value)} ${units}`}`,
            },
            {
                value: Math.round(values[0]),
                label: `${units === '$' ? `${units}${Math.round(values[0])}` : `${Math.round(values[0])} ${units}`}`,
            },
            {
                value: Math.round(values[1]),
                label: `${units === '$' ? `${units}${Math.round(values[1])}` : `${Math.round(values[1])} ${units}`}`,
            },
            {
                value: Math.round(feature.max),
                label: `${units === '$' ? `${units}${Math.round(feature.max)}` : `${Math.round(feature.max)} ${units}`}`,
            },
        ];
        return slider_marks;
    };
    const returnMinMax = (features, xid) => {
        const feature = features.find(feature => feature.xid === xid);
        return { min: feature.min, max: feature.max };
    };

    const getPercentChange = (currentValue, valueRange) => {
        const [minValue, maxValue] = valueRange;

        // Check if the current value is within the range
        if (currentValue >= minValue && currentValue <= maxValue) {
            return 0; // No change needed
        } else {
            // Calculate the distance from the current value to the min and max values
            const distanceToMin = Math.abs(currentValue - minValue);
            const distanceToMax = Math.abs(currentValue - maxValue);

            // Determine which distance is smaller and use that to calculate percent change
            const smallerDistance = distanceToMin < distanceToMax ? distanceToMin : distanceToMax;

            // Calculate percent change
            const percentChange = Math.round((smallerDistance / maxValue) * 100 * 10) / 10; // Round to 1 decimal place

            return percentChange;
        }
    };
    // % CHANGE:

    const ExplanationSliders = ({ sc1_explanations, sc2_explanations, sc1_explanations_index, sc2_explanations_index, sc1_features, sc2_features, }) => {
        if (sc1_explanations === undefined || sc2_explanations === undefined || sc1_explanations_index === undefined || sc2_explanations_index === undefined || sc1_features === undefined || sc2_features === undefined) {
            return null;
        }
        return (
            <div>
                <div style={{ marginLeft: '20px', width: '75%' }}>
                    {Object.entries(sc1_explanations[sc1_explanations_index]).map(([key, values], idx) => {
                        // scenario 1
                        const feature = sc1_features.find(feature => feature.xid === `x${idx}`);
                        const id = feature ? feature.title : '';
                        const units = feature ? feature.units : '';
                        const marks = createMarks(sc1_features, `x${idx}`, units, values);
                        const { min, max } = returnMinMax(sc1_features, `x${idx}`);
                        const percentChange1 = getPercentChange(feature.current_value, values);
                        // Scenario 2
                        const values2 = Object.values(sc2_explanations[sc2_explanations_index])[idx];
                        const marks2 = createMarks(sc2_features, key, units, values2);
                        const { min: min2, max: max2 } = returnMinMax(sc2_features, key);
                        const percentChange2 = getPercentChange(feature.current_value, values2); 
                        return (
                            <div key={idx} style={{}}>
                                <p className="slider-title" style={{ fontWeight: 'bold' }}>{id} ({units})</p>
                                <div className="slider-box">
                                    <p className="slider-id">{scenarioOne.scenarioID}</p>
                                    <StyledExplanationSlider
                                        className={`scenario-explanation-slider ${percentChange1 === 0 ? "No_Change" : ""}`}
                                        value={values}
                                        valueLabelDisplay="auto"
                                        getAriaValueText={(value) => rangeText(value)}
                                        min={min}
                                        max={max}
                                        marks={marks}
                                        disabled={true}
                                        style={{ marginTop: 20 }}
                                    />
                                    <p className={`percent-change ${percentChange1 < percentChange2 ? 'blue-text' : ''}`}>{percentChange1}%</p>
                                </div>
                                <div className="slider-box">
                                    <p className="slider-id">{scenarioTwo.scenarioID}</p>
                                    <StyledExplanationSlider
                                        className={`scenario-explanation-slider ${percentChange1 === 0 ? "No_Change" : ""}`}
                                        value={values2}
                                        valueLabelDisplay="auto"
                                        getAriaValueText={(value) => rangeText(value)}
                                        min={min2}
                                        max={max2}
                                        marks={marks2}
                                        disabled={true}
                                        style={{ marginTop: 20 }}
                                    />
                                    <p className={`percent-change ${percentChange2 < percentChange1 ? 'blue-text' : ''}`}>{percentChange2}%</p>
                                </div>

                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };
    return (
        <div className="compare-section">
            <h2>Comparison</h2>
            <div style={{ display: 'flex', flexDirection: 'row' }}>
                {/* Autocomplete for Scenario One */}
                <Autocomplete
                    className="dropdown-one"
                    options={savedScenarios.map((_, index) => `Scenario ${index}`)}
                    onChange={handleDropDownOne}
                    renderInput={(params) => (
                        <TextField
                            {...params}
                            label='Select Scenario'
                        />
                    )}
                    style={{ width: 200 }}
                />

                <Autocomplete
                    className="dropdown-one-explanation-index"
                    options={scenarioOne.explanations ? scenarioOne.explanations.map((_, index) => `${index}`) : []}
                    onChange={handleDropDownIndexOne}
                    renderInput={(params) => (
                        <TextField
                            {...params}
                            label='Index'
                        />
                    )}
                    selectOnFocus
                    clearOnBlur
                    handleHomeEndKeys
                    freeSolo
                    style={{ width: 80, marginLeft: 5 }}
                />
                {/* Autocomplete for Scenario OneExplanation Index*/}
                <Autocomplete
                    className="dropdown-two"
                    options={savedScenarios.map((_, index) => `Scenario ${index}`)}
                    onChange={handleDropDownTwo}
                    renderInput={(params) => (
                        <TextField
                            {...params}
                            label='Select Scenario'
                        />
                    )}
                    style={{ width: 200 }}
                />
                {/* Autocomplete for Scenario Two Explanation Index*/}
                <Autocomplete
                    className="dropdown-two-explanation-index"
                    options={scenarioTwo.explanations ? scenarioTwo.explanations.map((_, index) => `${index}`) : []}
                    onChange={handleDropDownIndexTwo}
                    renderInput={(params) => (
                        <TextField
                            {...params}
                            label='Index'
                        />
                    )}
                    selectOnFocus
                    clearOnBlur
                    handleHomeEndKeys
                    freeSolo
                    style={{ width: 80, marginLeft: 5 }}
                />
            </div>
            <div className="compare-container" style={{ maxHeight: 308, minHeight: 308, overflowY: 'auto' }}>
                <div className='sliders-comparison'>
                    <ExplanationSliders
                        sc1_explanations={scenarioOne.explanations}
                        sc2_explanations={scenarioTwo.explanations}
                        sc1_explanations_index={scenarioOneIndex}
                        sc2_explanations_index={scenarioTwoIndex}
                        sc1_features={scenarioOne.features}
                        sc2_features={scenarioTwo.features} />
                </div>
            </div>
        </div>
    );
};

export default ScenarioComparison;
