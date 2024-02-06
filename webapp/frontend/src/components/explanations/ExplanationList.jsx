import React, { useState } from 'react';
import NumberLine from './NumberLine';
import { formatFeature } from "../../js/utilities.js";

const ExplanationList = ({ explanations, totalExplanations, featureDict, formatDict }) => {
    const [currentExplanationIndex, setCurrentExplanationIndex] = useState(0);

    const handleNext = () => {
        setCurrentExplanationIndex((prevIndex) => (prevIndex + 1) % explanations.length);
    };

    const handlePrevious = () => {
        setCurrentExplanationIndex((prevIndex) => (prevIndex - 1 + explanations.length) % explanations.length);
    };

    return (
        <div className="explanations-container" style={{ margin: 10, flex: 1, overflowY: 'auto' }}>
            <div>
                {Object.keys(explanations[currentExplanationIndex]).map((key, innerIndex) => (
                    <div key={innerIndex} style={{ display: 'flex', flexDirection: 'row', overflow: 'auto' }}>
                        <h3 style={{ minWidth: 180 }}>
                            {formatFeature(key, formatDict)}
                        </h3>
                        <NumberLine
                            key={innerIndex}
                            explanation={totalExplanations[currentExplanationIndex]}
                            i={innerIndex}
                            id={`number-line-container-${currentExplanationIndex}-${innerIndex}`}
                        />
                    </div>
                ))}
            </div>

            {/* Next and Previous buttons */}
            {explanations.length > 1 && (
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10 }}>
                    <button onClick={handlePrevious} disabled={currentExplanationIndex === 0}>Previous</button>
                    <button onClick={handleNext} disabled={currentExplanationIndex === explanations.length - 1}>Next</button>
                </div>
            )}
        </div>
    );
};

export default ExplanationList;
