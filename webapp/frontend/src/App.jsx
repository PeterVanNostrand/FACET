import axios from 'axios';
import { useState, useEffect, useRef } from 'react'
import './css/App.css'
import NumberLine from './components/explanation/NumberLine';
import { autoType } from 'd3';

const multipleExplanations = 5

function App() {
    const [applications, setApplications] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedApplication, setSelectedApplication] = useState('');
    const [explanations, setExplanations] = useState([]);
    const [constraints, setConstraints] = useState([]);
    const [showForm, setShowForm] = useState(false)
    const [numExplanations, setNumExplanations] = useState(1)

    // useEffect to fetch applications data when the component mounts
    useEffect(() => {
        const fetchApplications = async () => {
            try {
                const response = await axios.get('http://localhost:3001/facet/applications');
                setApplications(response.data);
                setSelectedApplication(response.data[0]);
            } catch (error) {
                console.error(error);
            }

        };
        // Call the fetchApplications funtion when the component mounts
        fetchApplications();
        setConstraints([
            [1000, 1600],
            [0, 10],
            [6000, 10000],
            [300, 500]
        ])
    }, []);

    // useEffect to handle explanation when the selected application changes
    useEffect(() => {
        handleExplanation();
    }, [selectedApplication, numExplanations, constraints]);


    const featureDict = {
        "x0": "Applicant Income",
        "x1": "Coapplicant Income",
        "x2": "Loan Amount",
        "x3": "Loan Amount Term"
    }

    // Function to fetch explanation data from the server
    const handleExplanation = async () => {
        if (constraints.length === 0) return;

        try {
            const response = await axios.post(
                'http://localhost:3001/facet/explanation',
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

    // Function to handle displaying the previous application
    const handlePrevApp = () => {
        if (count > 0) {
            setCount(count - 1);
            setSelectedApplication(applications[count - 1]);
        }
        handleExplanation();
    }

    // Function to handle displaying the next application
    const handleNextApp = () => {
        if (count < applications.length - 1) {
            setCount(count + 1);
            setSelectedApplication(applications[count + 1]);
        }
        handleExplanation();
    }

    const handleNumExplanations = (numExplanations) => () => {
        setNumExplanations(numExplanations);
    }

    return (
        <div className="container" style={{ display: 'flex', flexDirection: 'row', height: '95vh', overflow: 'auto' }}>
            <div className='number-line-container'>
                <NumberLine />
            </div>

            <div
                className="applicant-container"
                style={{
                    position: 'sticky',
                    top: 0,
                }}
            >
                <h2>Application {count}</h2>
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

            <div className="explanation-container" style={{ marginLeft: 40, marginRight: 40 }}>
                {explanations.map((item, index) => (
                    <div key={index}>
                        <h2>Explanation {index + 1}</h2>
                        {Object.keys(item).map((key, innerIndex) => (
                            <div key={innerIndex}>
                                <h3>{featureDict[key]}</h3>
                                <p>{item[key][0]}, {item[key][1]}</p>
                                {/* <NumberLine explanationData={item[key]} /> */}
                            </div>
                        ))}
                        <p style={elementSpacer}></p>
                    </div >
                ))
                }
            </div>
        </div >
    )
}

const elementSpacer = {
    marginTop: 50,
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
