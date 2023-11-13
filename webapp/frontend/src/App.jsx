import axios from 'axios';
import { useState } from 'react'
import './css/App.css'

function App() {
    const [explanation, setExplanation] = useState('');

    const instance = {
        "ApplicantIncome": 4583,
        "CoapplicantIncome": 1508,
        "LoanAmount": 12800,
        "LoanAmountTerm": 360
    }

    const featureDict = {
        "x0": "Applicant Income",
        "x1": "Coapplicant Income",
        "x2": "Loan Amount",
        "x3": "Loan Amount Term"
    }

    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
            const response = await axios.post(
                'http://localhost:3001/facet/explanation',
                instance,
            );

            setExplanation(response.data);
        } catch (error) {
            console.error(error);
        }
    }

    return (
        <div>
            <div>
                <h2>Instance</h2>
                <p>Applicant Income: {instance.ApplicantIncome}</p>
                <p>Coapplicant Income: {instance.CoapplicantIncome}</p>
                <p>Loan Amount: {instance.LoanAmount}</p>
                <p>Loan Amount Term: {instance.LoanAmountTerm}</p>
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
