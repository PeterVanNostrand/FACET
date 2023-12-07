import React from 'react';
import ExplanationList from "./ExplanationList";

const ExplanationSection = ({ explanations, featureDict }) => {

    return (
        <ExplanationList explanations={explanations} featureDict={featureDict} />
    );
};

export default ExplanationSection;
