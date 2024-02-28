import { formatFeature } from "@src/js/utilities.js";
import NumberLine from './NumberLine';

const ExplanationSection = (
    { explanations,
        totalExplanations,
        formatDict,
        currentExplanationIndex,
        setCurrentExplanationIndex,
        saveScenario }
) => {

    const handleNext = () => {
        setCurrentExplanationIndex((prevIndex) => (prevIndex + 1) % explanations.length);
    };

    const handlePrevious = () => {
        setCurrentExplanationIndex((prevIndex) => (prevIndex - 1 + explanations.length) % explanations.length);
    };


    return (
        <div className="explanation-section" style={{
            display: 'flex', flexDirection: 'column', position: 'relative', minHeight: '100%'
        }}>
            <h2 className='explanation-header' style={{ marginBottom: 20, }}>
                Explanations
            </h2>

            {totalExplanations.length > 0 ? (
                <div className="explanation-container" style={{ maxHeight: 340 }}>
                    <div className="explanation-list">
                        {explanations.length > 0 && Object.keys(
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
                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                        {explanations.length > 1 && (
                            <div style={{
                                display: 'flex',
                                marginTop: 10,
                            }}>
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

            ) : (
                <i className="instructions-text">
                    No explanations available. Please relax the constraints to generate explanations.
                </i>
            )}

            {explanations.length != 0 && <button onClick={saveScenario} style={{
                marginTop: 15,
                width: 'fit-content'
            }}>
                Save Scenario
            </button>}

        </div>
    );
};

export default ExplanationSection;
