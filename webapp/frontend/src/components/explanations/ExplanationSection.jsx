import React, { useState } from 'react';
import ExplanationList from "./ExplanationList";

const ExplanationSection = (
    { explanations, totalExplanations, featureDict, formatDict, numExplanations, handleNumExplanations }
) => {

    return (
        <div className="explanation-section" style={{ display: 'flex', flexDirection: 'column' }}>
            <div className='explanation-sort' style={{ display: 'flex' }}>
                <button className={numExplanations === 1 ? "selected toggle" : "toggle"} onClick={handleNumExplanations(1)}>Top Explanation</button>
                <button className={numExplanations === 10 ? "selected toggle" : "toggle"} onClick={handleNumExplanations(10)}>Multiple Explanations</button>
            </div>

            <ExplanationList
                explanations={explanations}
                totalExplanations={totalExplanations}
                featureDict={featureDict}
                formatDict={formatDict}
            />
        </div>
    );
};

export default ExplanationSection;
