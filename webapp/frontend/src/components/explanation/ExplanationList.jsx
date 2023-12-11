import React from 'react';
import NumberLine from './NumberLine';

const ExplanationList = ({ explanations, totalExplanations, featureDict }) => {

    // const explanation = {
    //     "instance": {
    //         "x0": 4895,
    //         "x1": 0.0,
    //         "x2": 10200,
    //         "x3": 360
    //     },
    //     "region": {
    //         "x0": [
    //             412, 1013
    //         ],
    //         "x1": [
    //             0, 8
    //         ],
    //         "x2": [
    //             7100, 12350
    //         ],
    //         "x3": [
    //             48, 420
    //         ]
    //     }
    // };

    return (
        <div className="explanation-container" style={{ marginLeft: 40, marginRight: 40 }}>
            {/* iterate thru all explanations */}
            {explanations.map((item, outerIndex) => (
                <div key={outerIndex}>
                    <h2>Explanation {outerIndex + 1}</h2>

                    {/* iterature thru all features  */}
                    {Object.keys(item).map((key, innerIndex) => (
                        <div key={innerIndex} style={{ display: 'flex', flexDirection: 'row', overflow: 'auto' }}>
                            <h3 style={{ minWidth: 180, display: 'flex', justifyContent: 'flex-end' }}>{featureDict[key]}:</h3>
                            <NumberLine key={innerIndex} explanation={totalExplanations[outerIndex]} i={innerIndex} id={`number-line-container-${outerIndex}-${innerIndex}`} />
                            <p>{item[key][0]}, {item[key][1]}</p>
                        </div>
                    ))}

                    <p style={{ marginTop: 50 }}></p>
                </div >
            ))
            }

        </div >
    );
};

export default ExplanationList;
