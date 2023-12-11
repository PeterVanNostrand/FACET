import axios from 'axios';
import { useState, useEffect, useRef } from 'react'
import ExplanationSection from './components/my-application/explanation/ExplanationSection';
import { fetchApplications, fetchExplanations } from './js/api.js';
import NavBar from './components/NavBar';

const multipleExplanations = 10
const featureDict = {
    "x0": "Applicant Income",
    "x1": "Coapplicant Income",
    "x2": "Loan Amount",
    "x3": "Loan Amount Term"
}


function App() {
    const [applications, setApplications] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedApplication, setSelectedApplication] = useState('');
    const [constraints, setConstraints] = useState([]);
    const [numExplanations, setNumExplanations] = useState(1);
    const [explanations, setExplanations] = useState([]);
    const [totalExplanations, setTotalExplanations] = useState([]);
    const [explanationSection, setExplanationSection] = useState(null);

    // useEffect to fetch applications data when the component mounts
    useEffect(() => {
        fetchApplications();
        setConstraints([
            [1000, 1600],
            [0, 10],
            [6000, 10000],
            [300, 500]
        ])
    }, []);

    const fetchApplications = async () => {
        try {
            const response = await axios.get('http://localhost:3001/facet/applications');
            setApplications(response.data);
            setSelectedApplication(response.data[0]);
        } catch (error) {
            console.error(error);
        }

    };

    // Function to fetch explanation data from the server
    const handleExplanations = async () => {
        if (constraints.length === 0 || selectedApplication.length == 0) return;

        try {
            const response = await axios.post(
                'http://localhost:3001/facet/explanations',
                {
                    "num_explanations": numExplanations,
                    "x0": selectedApplication.x0,
                    "x1": selectedApplication.x1,
                    "x2": selectedApplication.x2,
                    "x3": selectedApplication.x3,
                    "constraints": constraints
                },
            );
            setExplanations(response.data);
        } catch (error) {
            console.error(error);
        }
    }

    // useEffect to handle explanation when the selected application changes
    useEffect(() => {
        handleExplanations();
    }, [selectedApplication, numExplanations, constraints]);

    useEffect(() => {
        const instanceAndExpls = explanations.map(region => ({
            instance: selectedApplication,
            region
        }));
        setTotalExplanations(instanceAndExpls);
    }, [explanations])

    useEffect(() => {
        if (totalExplanations.length > 0)
            setExplanationSection(
                <ExplanationSection
                    explanations={explanations}
                    totalExplanations={totalExplanations}
                    featureDict={featureDict}
                />
            )
    }, [totalExplanations])

    // Function to handle displaying the previous application
    const handlePrevApp = () => {
        if (count > 0) {
            setCount(count - 1);
            setSelectedApplication(applications[count - 1]);
        }
        handleExplanations();
    }

    // Function to handle displaying the next application
    const handleNextApp = () => {
        if (count < applications.length - 1) {
            setCount(count + 1);
            setSelectedApplication(applications[count + 1]);
        }
        handleExplanations();
    }

    const handleApplicationChange = (event) => {
        setCount(event.target.value);
        setSelectedApplication(applications[event.target.value]);
    };

    const handleNumExplanations = (numExplanations) => () => {
        setNumExplanations(numExplanations);
    }


    return (
        <div className='app-container'>
            <div className='filter-container'></div>
            <div className='feature-control-container'></div>
            <div className="my-application-container" style={{ overflow: 'auto', border: 'solid 1px red' }}>

                <div className="applicant-container" style={{}}>
                    <h2>My Application ({count})</h2>
                    
                    <select value={count} onChange={handleApplicationChange}>
                        {applications.map((applicant, index) => (
                            <option key={index} value={index}>
                                Application {index}
                            </option>
                        ))}
                    </select>

                    <button onClick={handlePrevApp}>Previous</button>
                    <button onClick={handleNextApp}>Next</button>

                    <p><em>Feature (Constraints): <span className="featureValue">Value</span></em></p>
                    {Object.keys(selectedApplication).map((key, index) => (
                        <div key={index}>
                            <Feature
                                name={featureDict[key]}
                                constraint={constraints[index]}
                                value={selectedApplication[key]}
                                updateConstraint={(i, newValue) => {
                                    const updatedConstraints = [...constraints];
                                    updatedConstraints[index][i] = newValue;
                                    setConstraints(updatedConstraints);
                                }}
                            />
                        </div>
                    ))}

                    <button onClick={handleNumExplanations(1)}>Single Explanation</button>
                    <button onClick={handleNumExplanations(multipleExplanations)}>List of Explanations</button>
                </div>

                {explanationSection}
            </div>
        </div>

    )
}


function Feature({ name, constraint, value, updateConstraint }) {
    return (
        <div className="features-container">
            <div className='feature'>
                <div>{name}&nbsp;
                    (<EditableText
                        currText={constraint[0]}
                        updateValue={(newValue) => updateConstraint(0, newValue)}
                    />,&nbsp;
                    <EditableText
                        currText={constraint[1]}
                        updateValue={(newValue) => updateConstraint(1, newValue)}
                    />)
                    : <span className="featureValue">{value}</span>
                </div>
            </div>
        </div>
    )
}


const EditableText = ({ currText, updateValue }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [text, setText] = useState(currText);

    const handleDoubleClick = () => {
        setIsEditing(true);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            setIsEditing(false);
            // Pass the updated value to the parent component
            updateValue(parseInt(text));
        }
    };

    const handleBlur = () => {
        setIsEditing(false);
        // Pass the updated value to the parent component
        updateValue(text);
    };

    const handleChange = (e) => {
        setText(e.target.value);
    };

    return (
        <div style={{ display: 'inline-block' }}>
            {isEditing ? (
                <input
                    style={{ width: Math.min(Math.max(text.length, 2), 20) + 'ch' }}
                    type="text"
                    value={text}
                    autoFocus
                    onChange={handleChange}
                    onBlur={handleBlur}
                    onKeyPress={handleKeyPress}
                />
            ) : (
                <p onClick={handleDoubleClick} style={{ cursor: 'pointer' }}>
                    {text}
                </p>
            )}
        </div>
    );
};

export default App;
