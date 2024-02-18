import React from 'react';
import { formatFeature } from "../../js/utilities.js";
import NumberLine from './NumberLine';

// Function component for rendering a single explanation item
const ExplanationItem = ({ explanation, innerIndex, formatDict }) => {
    console.log("Mapping explanation: ", explanation);
    return (
        <div key={innerIndex} style={{ display: 'flex', flexDirection: 'row' }}>
            <h3>
                {formatFeature(key, formatDict)}
            </h3>
            <NumberLine
                explanation={explanation}
                i={innerIndex}
                id={`number-line-container-${innerIndex}`}
            />
        </div>
    );
};

// Function component for rendering the list of explanations
const ExplanationList = ({ explanations, explanationIndex, formatDict }) => {
    // Check if explanations[explanationIndex] exists and is an object
    if (explanations && Array.isArray(explanations) && explanations.length > explanationIndex) {
        const explanation = explanations[explanationIndex];

        return (
            <div className="explanation-list">
                {Object.keys(explanation).map((key, innerIndex) => {
                    const explanationValues = explanation[key];
                    return (
                        <div key={innerIndex}>
                            <h3>{formatFeature(key, formatDict)}</h3>
                            <NumberLine
                                explanation={explanationValues}
                                i={innerIndex}
                                id={`number-line-container-${innerIndex}`}
                            />
                        </div>
                    );
                })}
            </div>
        );
    } else {
        return null; // Render nothing if explanations[explanationIndex] is not an object or doesn't exist
    }
};


const ScenarioComparison = ({ savedScenarios, scenario_1, scenario_2, formatDict}) => {
    const scenarios_length = savedScenarios.length;

    if (scenarios_length >= 2) {
        const scenario1 = savedScenarios[scenario_1];
        const explanation_length = scenario1.explanations.length;

        return (
            <div className="compare-section" style={{ display: 'flex', flexDirection: 'column', position: 'relative', minHeight: '100%' }}>
                <h2>Comparison</h2>

                <div className="compare-container" style={{ maxHeight: 308, minHeight: 308, position: 'relative' }}>
                    <ExplanationList explanations={scenario1.explanations} explanationIndex={scenario1.explanationIndex} formatDict={formatDict} />
                </div>
                <div>
                    {/* Add other components or elements */}
                </div>
            </div>
        );
    } else {
        return null; // Render nothing if there are not enough scenarios
    }
};

export default ScenarioComparison;
