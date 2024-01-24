import React from 'react';

const TempWelcome = ({ setIsWelcome, handleApplicationChange, count, instances }) => {


    return (
        <div>
            <h1>Welcome!</h1>
            <p>Enjoy your stay.</p>
            <select value={count} onChange={handleApplicationChange}>
                {instances.map((applicant, index) => (
                    <option key={index} value={index}>
                        Application {index}
                    </option>
                ))}
            </select>
            <button
                onClick={() => setIsWelcome(false)}
            >
                Take me to my application!
            </button>
        </div>
    );
};

export default TempWelcome;
