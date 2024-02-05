import React, { useState } from 'react';
import NumberLine from './NumberLine';

const ExplanationList = ({ explanations, totalExplanations, featureDict }) => {
  const [currentExplanationIndex, setCurrentExplanationIndex] = useState(0);

  const handleNext = () => {
    setCurrentExplanationIndex((prevIndex) => (prevIndex + 1) % explanations.length);
  };

  const handlePrevious = () => {
    setCurrentExplanationIndex((prevIndex) => (prevIndex - 1 + explanations.length) % explanations.length);
  };

  return (
    <div className="explanations-container" style={{ margin: 10, minHeight: '35vh', flex: 1, overflowY: 'auto' }}>
      {/* Display the current explanation */}
      <div>
        {/* iterate thru all features  */}
        {Object.keys(explanations[currentExplanationIndex]).map((key, innerIndex) => (
          <div key={innerIndex} style={{ display: 'flex', flexDirection: 'row', overflow: 'auto' }}>
            <h3 style={{ minWidth: 180, display: 'flex', justifyContent: 'flex-end' }}>{featureDict[key]}</h3>
            <NumberLine key={innerIndex} explanation={totalExplanations[currentExplanationIndex]} i={innerIndex} id={`number-line-container-${currentExplanationIndex}-${innerIndex}`} />
          </div>
        ))}
        <p style={{ marginTop: 40 }}></p>
      </div>

      {/* Next and Previous buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10 }}>
        <button onClick={handlePrevious} disabled={currentExplanationIndex === 0}>Previous</button>
        <button onClick={handleNext} disabled={currentExplanationIndex === explanations.length - 1}>Next</button>
      </div>
    </div>
  );
};

export default ExplanationList;
