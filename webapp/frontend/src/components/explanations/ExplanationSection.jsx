import React, { useState } from 'react';
import ExplanationList from "./ExplanationList";

const ExplanationSection = (
    { explanations, totalExplanations, featureDict, formatDict, numExplanations, handleNumExplanations }
) => {

    return (
        <div className="explanation-section" style={{
            display: 'flex', flexDirection: 'column'
        }}>
            <div className='explanation-sort' style={{ display: 'flex' }}>
                <button className={numExplanations === 1 ? "selected toggle" : "toggle"} onClick={handleNumExplanations(1)}>Top Explanation</button>
                <button className={numExplanations === 10 ? "selected toggle" : "toggle"} onClick={handleNumExplanations(10)}>Multiple Explanations</button>
            </div>
            <div style={{
                display: 'flex', 
                flexDirection: 'column', 
                boxShadow: "inset 0 0 10px 5px rgba(0, 0, 0, 0.03)",
                marginBottom: "10px",
                marginTop: "10px",
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
