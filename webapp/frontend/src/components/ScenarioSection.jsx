import React, { useState } from 'react';
import close from '../../icons/close.svg';

const ScenarioSection = ({ savedScenarios, setSavedScenarios, setCurrentExplanationIndex, setSelectedInstance }) => {
    const [selectedScenarioIndex, setSelectedScenarioIndex] = useState(null);

    const handleScenarioChange = (scenario, index) => {
        setSelectedInstance(scenario.values)
        setCurrentExplanationIndex(scenario.explanationIndex);
        setSelectedScenarioIndex(index);
    }

    const deleteScenario = (index) => {
        let newScenarios = savedScenarios.slice();
        newScenarios.splice(index, 1);
        setSavedScenarios(newScenarios);
        if (selectedScenarioIndex === index) {
            setSelectedScenarioIndex(null);
        }
    }

    const clearScenarios = () => {
        setSavedScenarios([]);
        setSelectedScenarioIndex(null); // Reset selected index when scenarios are cleared
    }

    return (
        <div>
            <h2>Scenarios</h2>
            {savedScenarios.length == 0
                ?
                <i style={{ color: '#777777' }}>Save an explanation to create a scenario</i>
                :
                <div className="tab-list">
                    {savedScenarios.map((scenario, index) => (
                        <div
                            key={index}
                            className={`tab ${selectedScenarioIndex === index ? 'selected' : ''}`}
                            onClick={() => { handleScenarioChange(scenario, index) }}
                        >
                            <p>
                                Scenario {scenario.scenarioID}.{scenario.explanationIndex + 1}
                            </p>
                            <button className="tab-close" onClick={() => deleteScenario(index)}>
                                <img src={close} alt="close" />
                            </button>
                        </div>
                    ))}
                </div>
            }
        </div>
    );
};

export default ScenarioSection;
