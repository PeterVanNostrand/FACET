import axios from 'axios';
import { useState, useEffect } from 'react'
import './css/App.css'


function App() {
    const [applications, setApplications] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedApplication, setSelectedApplication] = useState('');
    const [explanation, setExplanation] = useState('');

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
        fetchApplications();
    }, []);

    useEffect(() => {
        handleExplanation()
    }, [selectedApplication]);

    const featureDict = {
        "x0": "Applicant Income",
        "x1": "Coapplicant Income",
        "x2": "Loan Amount",
        "x3": "Loan Amount Term"
    }

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

            setExplanation(response.data);
        } catch (error) {
            console.error(error);
        }
    }

    const handlePrevApp = () => {
        if (count > 0) {
            setCount(count - 1);
            setSelectedApplication(applications[count - 1]);
        }
        handleExplanation();
    }

    const handleNextApp = () => {
        if (count < applications.length - 1) {
            setCount(count + 1);
            setSelectedApplication(applications[count + 1]);
        }
        handleExplanation();
    }

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

            {Object.keys(explanation).map((key, index) => (
                <div key={index}>
                    <h3>{featureDict[key]}</h3>
                    <p>{explanation[key][0]}, {explanation[key][1]}</p>
                </div>
            ))}
        </>
    )
}

export default App
