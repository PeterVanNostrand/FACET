import React from 'react';
import NumberLine from './NumberLine';

const ExplanationList = ({ explanations, featureDict }) => {

    const explanation = {
        "instance": {
            "x0": 4895,
            "x1": 0.0,
            "x2": 10200,
            "x3": 360
        },
        "region": {
            "x0": [
                412, 1013
            ],
            "x1": [
                0, 8
            ],
            "x2": [
                7100, 12350
            ],
            "x3": [
                48, 420
            ]
        }
    };

    return (
        <div className="explanation-container" style={{ marginLeft: 40, marginRight: 40 }}>

            {explanations.map((item, index) => (
                <div key={index}>
                    <h2>Explanation {index + 1}</h2>

                    <div id={`number-line-container-${index}`}>
                        {[...Array(4)].map((_, i) => (
                            <NumberLine key={i} explanation={explanation} i={i} id={`number-line-container-${index}`} />
                        ))}
                    </div>

                    {Object.keys(item).map((key, innerIndex) => (
                        <div key={innerIndex}>
                            <h3>{featureDict[key]}</h3>
                            <p>{item[key][0]}, {item[key][1]}</p>
                            {/* <NumberLine explanationData={item[key]} /> */}
                        </div>
                    ))}

                    <p style={{ marginTop: 50 }}></p>
                </div >
            ))}

        </div>
    );
};

export default ExplanationList;
