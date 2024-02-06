import React, { useState } from 'react';
import ExplanationList from "./ExplanationList";

const ExplanationSection = (
    { explanations, totalExplanations, featureDict, formatDict, numExplanations, handleNumExplanations }
) => {

    return (
        <div className="explanation-section" style={{
            display: 'flex', flexDirection: 'column'
        }}>
            <h2 className='explanation-header'>
                Explanations
            </h2>
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                boxShadow: "inset 0 0 10px 5px rgba(0, 0, 0, 0.03)",
                marginBottom: "15px",
                paddingLeft: "15px",
                paddingRight: "10px",
                borderRadius: "10px",
            }}>
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
