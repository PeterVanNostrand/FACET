import React from 'react';
import { formatFeature } from "../../js/utilities.js";
import NumberLine from './NumberLine';

const ExplanationSection = (
    { explanations, totalExplanations, formatDict, currentExplanationIndex, setCurrentExplanationIndex }
) => {

    const handleNext = () => {
        setCurrentExplanationIndex((prevIndex) => (prevIndex + 1) % explanations.length);
    };

    const handlePrevious = () => {
        setCurrentExplanationIndex((prevIndex) => (prevIndex - 1 + explanations.length) % explanations.length);
    };


    return (
        <div className="explanation-section" style={{
            display: 'flex', flexDirection: 'column'
        }}>
            <h2 className='explanation-header' style={{ marginBottom: 10 }}>
                Explanations
            </h2>

            <div className="explanation-container">
                <div className="explanation-list" >
                    {Object.keys(
                        explanations[explanations.length === 1 ? 0 : currentExplanationIndex]
                    ).map((key, innerIndex) => (
                        <div key={innerIndex} style={{ display: 'flex', flexDirection: 'row' }}>
                            <h3 >
                                {formatFeature(key, formatDict)}
                            </h3>
                            <NumberLine
                                key={innerIndex}
                                explanation={totalExplanations[explanations.length === 1 ? 0 : currentExplanationIndex]}
                                i={innerIndex}
                                id={`number-line-container-${currentExplanationIndex}-${innerIndex}`}
                            />
                        </div>
                    ))}
                </div>

                {/* Next and Previous buttons */}
                {explanations.length > 1 && (
                    <div style={{ display: 'flex', justifyContent: 'center', marginTop: 10 }}>
                        <button
                            className="cycle-button"
                            onClick={handlePrevious}
                            disabled={currentExplanationIndex === 0}
                        >
                            &lt;
                        </button>
                        <p style={{ width: 30, textAlign: 'center' }}>{'\u00A0'}{'\u00A0'}{currentExplanationIndex + 1}{'\u00A0'}{'\u00A0'}</p>
                        <button
                            className="cycle-button"
                            onClick={handleNext}
                            disabled={currentExplanationIndex === explanations.length - 1}
                        >
                            &gt;
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ExplanationSection;
