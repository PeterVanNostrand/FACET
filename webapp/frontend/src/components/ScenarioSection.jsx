import React from 'react';
import close from '../../icons/close.svg';

const ScenarioSection = ({ savedScenarios, setSavedScenarios, setCurrentExplanationIndex, setSelectedInstance }) => {

    const handleScenarioChange = (scenario) => {
        console.log("Scenario changed");
        setSelectedInstance(scenario.values)
        setCurrentExplanationIndex(scenario.explanationIndex);
    }

    const deleteScenario = (index) => {
        let newScenarios = savedScenarios.slice();
        newScenarios.splice(index, 1);
        setSavedScenarios(newScenarios);
    }

    const clearScenarios = () => {
        setSavedScenarios([]);
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
                        <div key={index} className="tab" onClick={() => { handleScenarioChange(scenario) }}>
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
