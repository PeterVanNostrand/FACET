import axios from 'axios';
import { useState, useEffect } from 'react'
import './css/App.css'

function App() {
    // State for storing applications data
    const [applications, setApplications] = useState([]);
    // State for tracking the current application index
    const [count, setCount] = useState(0);
    // State for storing the selected application
    const [selectedApplication, setSelectedApplication] = useState('');
    // State for storing the explanation data
    const [explanation, setExplanation] = useState('');

    // useEffect to fetch applications data when the component mounts
    useEffect(() => {
        const fetchApplications = async () => {
            try {
                // Fetch applications data from the server
                const response = await axios.get('http://localhost:3001/facet/applications');
                // Set the applications data in the state
                setApplications(response.data);
                // Set the selected application to the first application in the list
                setSelectedApplication(response.data[0]);
            } catch (error) {
                console.error(error);
            }
        };
        // Call the fetchApplications function when the component mounts
        fetchApplications();
    }, []);

    // useEffect to handle explanation when the selected application changes
    useEffect(() => {
        handleExplanation();
    }, [selectedApplication]);

    // Dictionary to map feature names to more readable names
    const featureDict = {
        "x0": "Applicant Income",
        "x1": "Coapplicant Income",
        "x2": "Loan Amount",
        "x3": "Loan Amount Term"
    }

    // Function to fetch explanation data from the server
    const handleExplanation = async () => {
        try {
            const response = await axios.post(
                'http://localhost:3001/facet/explanation',
                {
                    "x0": selectedApplication.x0,
                    "x1": selectedApplication.x1,
                    "x2": selectedApplication.x2,
                    "x3": selectedApplication.x3
                },
            );
            // Set the explanation data in the state
            setExplanation(response.data);
        } catch (error) {
            console.error(error);
        }
    }

    // Function to handle displaying the previous application
    const handlePrevApp = () => {
        if (count > 0) {
            // Decrease the count to move to the previous application
            setCount(count - 1);
            // Set the selected application to the previous application
            setSelectedApplication(applications[count - 1]);
        }
        // Fetch explanation data for the new selected application
        handleExplanation();
    }

    // Function to handle displaying the next application
    const handleNextApp = () => {
        if (count < applications.length - 1) {
            // Increase the count to move to the next application
            setCount(count + 1);
            // Set the selected application to the next application
            setSelectedApplication(applications[count + 1]);
        }
        // Fetch explanation data for the new selected application
        handleExplanation();
    }

    // JSX content to render
    return (
        <>
            <div>
                <h2>Application {count}</h2>
                <button onClick={handlePrevApp}>Previous</button>
                <button onClick={handleNextApp}>Next</button>

                <p>Applicant Income: {selectedApplication.x0}</p>
                <p>Coapplicant Income: {selectedApplication.x1}</p>
                <p>Loan Amount: {selectedApplication.x2}</p>
                <p>Loan Amount Term: {selectedApplication.x3}</p>
            </div>

            <h2>Explanation</h2>

            {/* Render explanation data */}
            {Object.keys(explanation).map((key, index) => (
                <div key={index}>
                    <h3>{featureDict[key]}</h3>
                    <p>{explanation[key][0]}, {explanation[key][1]}</p>
                </div>
            ))}
        </>
    )
}

export default App;
