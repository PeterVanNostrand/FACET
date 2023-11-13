import axios from 'axios';
import { useState, useEffect } from 'react'
import './css/App.css'


function App() {
    const [explanation, setExplanation] = useState('');
    const [applications, setApplications] = useState([]);
    const [selectedApplication, setSelectedApplication] = useState('');
    const [count, setCount] = useState(0);

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

    const featureDict = {
        "x0": "Applicant Income",
        "x1": "Coapplicant Income",
        "x2": "Loan Amount",
        "x3": "Loan Amount Term"
    }

    const handlePrevApp = () => {
        if (count > 0) {
            setCount(count - 1);
            setSelectedApplication(applications[count - 1]);
        }
    }

    const handleNextApp = () => {
        if (count < applications.length - 1) {
            setCount(count + 1);
            setSelectedApplication(applications[count + 1]);
        }
    }

    const handleSubmit = async (event) => {
        event.preventDefault();
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

    return (
        <div>
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

            <form onSubmit={handleSubmit}>
                <button type="submit">Get explanation</button>
            </form>

            {explanation && typeof explanation === 'object' && (
                Object.keys(explanation).map((key, index) => (
                    <div key={index}>
                        <h3>{featureDict[key]}</h3>
                        <p>{explanation[key][0]}, {explanation[key][1]}</p>
                    </div>
                ))
            )}
        </div>
    )
}

export default App
