import React, { useState } from 'react';
import ExplanationList from "./ExplanationList";

const ExplanationSection = (
    { explanations, totalExplanations, featureDict, formatDict, numExplanations, handleNumExplanations }
) => {

    return (
        <div className="explanation-section" style={{
            display: 'flex', flexDirection: 'column'
        }}>
            <h2 className='explanation-header' style={{marginBottom: 10}}>
                Explanations
            </h2>
            <div className="explanation-list">
                <ExplanationList
                    explanations={explanations}
                    totalExplanations={totalExplanations}
                    featureDict={featureDict}
                    formatDict={formatDict}
                />
            </div>
        </div>
    );
};

export default ExplanationSection;
