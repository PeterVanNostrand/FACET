import React from 'react';
import ExplanationList from "./ExplanationList";

const ExplanationSection = ({ explanations, totalExplanations, featureDict, handleNumExplanations }) => {

    return (
        <div className="explanation-section" style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="unscrollable" style={{ flex: '0 0 auto' }}>

                <div className='explanation-sort' style={{ display: 'flex' }}>
                    <button onClick={handleNumExplanations(1)}>Top Explanation</button>
                    <button onClick={handleNumExplanations(10)}>Multiple Explanations</button>
                </div>

            </div>

            <ExplanationList
                explanations={explanations}
                totalExplanations={totalExplanations}
                featureDict={featureDict}
            />
        </div>
    );
};

export default ExplanationSection;
