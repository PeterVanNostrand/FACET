import React from 'react';
import NumberLine from './NumberLine';
import Button from '@mui/material/Button';

const ScenarioComparison = ({ savedScenarios, scenario_1, scenario_2 }) => {
    // Retrieve scenarios based on their IDs
    const scenario1 = savedScenarios[scenario_1];
    const scenario2 = savedScenarios[scenario_2];

    // Function to render number lines for each feature in an explanation
    const renderNumberLines = (explanation, explanation_index) => {
        return Object.keys(explanation).map((feature, index) => (
            <NumberLine
                key={index}
                explanation={explanation}
                feature={feature}
                explanation_index={explanation_index}
            />
        ));
    };

    const renderComparison = (scenario1, scenario2) => {
        console.log("Print Information: ");
        const farts = savedScenarios[scenario_1];
        console.log("Saved Scenario 1: ", farts);
        console.log("explanations : ", farts.explanation);
        console.log("Saved Scenario 2: ", savedScenarios[scenario_2]);


    }

    // JSX to render comparison of scenarios
    return (
        <div>
            <h2>Comparison</h2>
            <div>
                <Button onClick={renderComparison}></Button>
            </div>
        </div>
    );
};

export default ScenarioComparison;
