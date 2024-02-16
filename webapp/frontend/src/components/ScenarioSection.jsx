import React, { useEffect } from 'react';
import close from '../../icons/close.svg';

const ScenarioSection = (
    { savedScenarios,
        setSavedScenarios,
        setExplanations,
        setCurrentExplanationIndex,
        setSelectedInstance,
        selectedScenarioIndex,
        setSelectedScenarioIndex,
        setConstraints
    }) => {

    const handleScenarioChange = (scenario, index) => {
        if (selectedScenarioIndex === index) return;

        setSelectedScenarioIndex(index);
    }

    useEffect(() => {
        if (selectedScenarioIndex !== null) {
            const scenario = savedScenarios[selectedScenarioIndex];

            setExplanations(scenario.explanations);
            setCurrentExplanationIndex(scenario.explanationIndex);
            setSelectedInstance(scenario.instance);
            setConstraints(scenario.constraints);
        }
    }, [selectedScenarioIndex]);

    const deleteScenario = (index, event) => {
        event.stopPropagation(); // prevent scenario change

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
        <div id="scenario-grid" className="card">
            <div style={{ display: 'flex' }}>
                <h2>Scenarios</h2>
                <p className="clear-scenarios" onClick={clearScenarios}>Clear All</p>
            </div>
            {savedScenarios.length == 0
                ?
                <i className="instructionsText">Save an explanation to create a scenario</i>
                :
                <div className="tab-list">
                    {savedScenarios.map((scenario, index) => (
                        <div
                            key={index}
                            className={`tab ${selectedScenarioIndex === index ? 'selected' : ''}`}
                            onClick={() => { handleScenarioChange(scenario, index) }}
                        >
                            <p>
                                Scenario {scenario.scenarioID}
                            </p>
                            <button className="tab-close-scenario" onClick={(event) => deleteScenario(index, event)}>
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
