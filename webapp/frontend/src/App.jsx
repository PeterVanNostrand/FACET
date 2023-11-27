import axios from 'axios';
import { useState, useEffect, useRef } from 'react'
import ExplanationSection from './components/my-application/explanation/ExplanationSection';
import { fetchApplications, fetchExplanations } from './js/api.js';
import NavBar from './components/NavBar';
import FeatureControlSection from './components/feature-control/FeatureControlSection';

import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';


const featureDict = {
    "x0": "Applicant Income",
    "x1": "Coapplicant Income",
    "x2": "Loan Amount",
    "x3": "Loan Amount Term"
}

import masterJson from '../../backend/visualization/data/dataset_details.json'

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
                    handleNumExplanations={handleNumExplanations}
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

    // refactorable section to handle adding a new profile
    // ---------------------------------------------------
    const [value, setValue] = useState(0);

    const handleChange = (event, newValue) => {
        setValue(newValue);
    };

    const handleAddProfile = () => {
        console.log('Add new profile logic here');
    };
    // ---------------------------------------------------


    return (
        <div className='main-container' style={{maxHeight: '98vh',}}>

            <div className="nav-bar"></div>
            <div className='app-body-container' style={{ display: 'flex', flexDirection: 'row' }}>
                <div className='filter-container'></div>
                <div className='feature-control-container' style={{ border: 'solid 1px black', padding: 20 }}>
                    <div style={{ maxHeight: 40 }}>

                        <FeatureControlSection />
                    </div>
                </div>

                <div className="my-application-container" style={{ overflowY: 'auto', border: 'solid 1px black', padding: 10 }}>
                    <div className='rhs' style={{ padding: 10 }}>

                        <h2 className='applicant-header' style={{ marginTop: 10, marginBottom: 20 }}>My Application</h2>
                        <select value={count} onChange={handleApplicationChange}>
                            {applications.map((applicant, index) => (
                                <option key={index} value={index}>
                                    Application {index}
                                </option>
                            ))}
                        </select>
                        <div className='applicant-tabs' style={{ display: 'flex', flexDirection: 'row' }}>
                            <Tabs value={1} onChange={handleChange} indicatorColor="primary">
                                <Tab label="Default" />
                                <Tab label={`Profile ${count}`} />
                            </Tabs>

                            <button style={{ border: '1px solid black', color: 'black', backgroundColor: 'white' }} onClick={handleAddProfile}>+</button>
                        </div>

                        <div className='applicant-info-container' style={{ margin: 10 }}>
                            {Object.keys(selectedApplication).map((key, index) => (
                                <div key={index} className='feature' style={{ margin: -10 }}>
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
                        </div>
                    </div>



                    {explanationSection}
                    <div className="suggestions-container">
                        <div>
                            <h2 style={{ marginTop: 10, marginBottom: 10 }}>Suggestions</h2>
                            
                        </div>
                        <p>
                            Your application would have been accepted if your income was $1,013-$1,519 instead of $4,895
                            and your loan was $9,450-$10,000 instead of $10,200.
                        </p>
                    </div>
                </div>

            </div>
        </div>

    )
}


function Feature({ name, constraint, value, updateConstraint }) {
    return (
        <div>
            <strong>{name}</strong>&nbsp;
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
