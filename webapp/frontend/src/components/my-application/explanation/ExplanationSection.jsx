import React from 'react';
import ExplanationList from "./ExplanationList";
import '../../../css/explanation-section.css';

const ExplanationSection = ({ explanations, totalExplanations, featureDict, handleNumExplanations }) => {

    return (
        <div className="explanation-section" style={{ padding: 10, borderTop: '3px solid #aaaaaa', display: 'flex', flexDirection: 'column' }}>
            <div className="unscrollable" style={{ flex: '0 0 auto' }}>

                <h2 className='explanation-header' style={{marginTop: 10, marginBottom: 10 }}>
                    Explanation(s)
                </h2>

                <div className='explanation-sort' style={{ display: 'flex' }}>
                    <p>Sort by:</p>
                    <button onClick={handleNumExplanations(1)}>Top Explanation</button>
                    <button onClick={handleNumExplanations(10)}>List</button>
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
