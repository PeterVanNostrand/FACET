import React from 'react';
import NumberLine from './NumberLine';

const ExplanationList = ({ explanations, totalExplanations, featureDict }) => {

    return (
        <div className="explanations-container" style={{ borderTop: '2px gray solid', borderBottom: '2px gray solid', margin: 10, minHeight: '35vh', maxHeight: '35vh', flex: 1, overflowY: 'auto' }}>
            {/* iterate thru all explanations */}
            {explanations.map((item, outerIndex) => (
                <div key={outerIndex}>

                    {/* iterature thru all features  */}
                    {Object.keys(item).map((key, innerIndex) => (
                        <div key={innerIndex} style={{ display: 'flex', flexDirection: 'row', overflow: 'auto' }}>
                            <h3 style={{ minWidth: 180, display: 'flex', justifyContent: 'flex-end' }}>{featureDict[key]}</h3>
                            <NumberLine key={innerIndex} explanation={totalExplanations[outerIndex]} i={innerIndex} id={`number-line-container-${outerIndex}-${innerIndex}`} />
                        </div>
                    ))}

                    <p style={{ marginTop: 40 }}></p>
                </div >
            ))
            }

        </div >
    );
};

export default ExplanationList;
