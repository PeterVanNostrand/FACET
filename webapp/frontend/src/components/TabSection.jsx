import React from 'react';
import close from '../../icons/close.svg';

const TabSection = ({ savedScenarios, deleteScenario, setSelectedInstance }) => {
    return (
        <div>
            <h2>Tabs</h2>
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
        </div>
    );
};

export default TabSection;
