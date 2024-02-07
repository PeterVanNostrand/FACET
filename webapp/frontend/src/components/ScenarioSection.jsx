import React from 'react';
import close from '../../icons/close.svg';

const ScenarioSection = ({ savedScenarios, setSavedScenarios, selectedInstance, setSelectedInstance }) => {

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
                <i style={{color: '#777777'}}>Save an explanation to create a scenario</i>
                :
                <div className="tab-list">
                    {savedScenarios.map((scenario, index) => (
                        <div key={index} className="tab" onClick={() => setSelectedInstance(scenario.values)}>
                            <p>
                                Scenario {scenario.scenarioID}
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
