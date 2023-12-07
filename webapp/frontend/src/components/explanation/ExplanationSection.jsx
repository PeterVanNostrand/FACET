import React from 'react';
import ExplanationList from "./ExplanationList";

const ExplanationSection = ({ explanations, totalExplanations, featureDict }) => {

    return (
        <ExplanationList
            explanations={explanations}
            totalExplanations={totalExplanations}
            featureDict={featureDict}
        />
    );
};

export default ExplanationSection;
