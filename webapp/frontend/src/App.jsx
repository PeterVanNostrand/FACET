import axios from 'axios';
import { useState, useEffect, useRef } from 'react'
import './css/App.css'
import NumberLine from './NumberLine';
import { autoType } from 'd3';

const multipleExplanations = 5

function App() {
    const [applications, setApplications] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedApplication, setSelectedApplication] = useState('');
    const [explanations, setExplanations] = useState([]);
    const [constraints, setConstraints] = useState([]);
    const [showForm, setShowForm] = useState(false)

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
        handleExplanation(multipleExplanations);
    }, [selectedApplication]);

    const featureDict = {
        "x0": "Applicant Income",
        "x1": "Coapplicant Income",
        "x2": "Loan Amount",
        "x3": "Loan Amount Term"
    }

    // Function to fetch explanation data from the server
    const handleExplanation = async (numExplanations) => {
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
        handleExplanation(1);
    }

    // Function to handle displaying the next application
    const handleNextApp = () => {
        if (count < applications.length - 1) {
            setCount(count + 1);
            setSelectedApplication(applications[count + 1]);
        }
        handleExplanation(1);
    }

    const handleNumExplanations = (numExplanations) => () => {
        handleExplanation(numExplanations);
    }

    return (
        <>
            <div>
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
                        />
                    </div>
                ))}

                <button onClick={handleNumExplanations(1)}>Single Explanation</button>
                <button onClick={handleNumExplanations(multipleExplanations)}>List of Explanations</button>
            </div>

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
        </>
    )
}

const elementSpacer = {
    marginTop: 50,
}


function Feature({ name, constraint, value }) {
    return (
        <div className="features-container">
            <div className='feature'>
                <div>{name}&nbsp;
                    (<EditableText currText={constraint[0]} />,&nbsp;
                    <EditableText currText={constraint[1]} />)
                    : <span className="featureValue">{value}</span>
                </div>
            </div>
        </div>
    )
}

const EditableText = ({ currText }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [text, setText] = useState(currText);

    const handleDoubleClick = () => {
        setIsEditing(true);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            setIsEditing(false);
        }
    };

    const handleBlur = () => {
        setIsEditing(false);
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
