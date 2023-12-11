import axios from 'axios';
import { useState, useEffect } from 'react'
//import './css/App.css'
//import './css/featurecontrolstyle.css'
import FeatureControlTab from './FeatureControlSection';
import './css/feature-control.css';
import NumberLine from './NumberLine';

const multipleExplanations = 5

function App() {
    const [applications, setApplications] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedApplication, setSelectedApplication] = useState('');
    const [explanations, setExplanations] = useState([]);

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
                    "x3": selectedApplication.x3
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
        <FeatureControlTab />
        {/* <div>
                <h2>Application {count}</h2>
                <button onClick={handlePrevApp}>Previous</button>
                <button onClick={handleNextApp}>Next</button>

                {Object.keys(selectedApplication).map((key, index) => (
                    <div key={index}>
                        <Feature name={featureDict[key]} value={selectedApplication[key]} />
                    </div>
                ))}

            </div>

            <button onClick={handleNumExplanations(1)}>Single Explanation</button>
            <button onClick={handleNumExplanations(multipleExplanations)}>List of Explanations</button>

            {explanations.map((item, index) => (
                <div key={index}>
                    <h3>{featureDict[key]}</h3>
                    <p>{explanation[key][0]}, {explanation[key][1]}</p>
                </div>
            ))} */}
        </>
    )
}

const elementSpacer = {
    marginTop: 80,
}


function Feature({ name, value }) {

    return (
        <div className="features-container">
            <div className='feature'>
                <p>{name}: <span className="featureValue">{value}</span></p>
            </div>
        </div>
    )
}

export default App;
