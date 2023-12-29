import axios from "axios";
import { useEffect, useState, useRef } from "react";
import webappConfig from "../../config.json";
import { formatFeature, formatValue } from "../utilities";

import ExplanationSection from './components/my-application/explanation/ExplanationSection';
import NavBar from './components/NavBar';
import FeatureControlSection from './components/feature-control/FeatureControlSection';

import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import { autoType, select } from "d3";

const success = "LimeGreen"
const failure = "Red"
function status_log(text, color) {
    if (color === null) {
        console.log(text)
    }
    else {
        console.log("%c" + text, "color:" + color + ";font-weight:bold;")
    }
}

function App() {
    const [instances, setInstances] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedInstance, setSelectedInstance] = useState("");
    const [explanations, setExplanations] = useState("");
    const [constraints, setConstraints] = useState([]);
    const [numExplanations, setNumExplanations] = useState(1);
    const [formatDict, setFormatDict] = useState(null);
    const [featureDict, setFeatureDict] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const [totalExplanations, setTotalExplanations] = useState([]);
    const [explanationSection, setExplanationSection] = useState(null);


    // determine the server path
    const SERVER_URL = webappConfig.SERVER_URL
    const API_PORT = webappConfig.API_PORT
    const ENDPOINT = SERVER_URL + ":" + API_PORT + "/facet"

    // useEffect to fetch instances data when the component mounts
    useEffect(() => {
        status_log("Using endpoint " + ENDPOINT, success)

        setConstraints([
            [1000, 1600],
            [0, 10],
            [6000, 10000],
            [300, 500]
        ])

        const pageLoad = async () => {
            fetchInstances();
            let dict_data = await fetchHumanFormat()
            setFormatDict(dict_data)
            setFeatureDict(dict_data["feature_names"]);
            setIsLoading(false);
        }

        // get the hman formatting data instances
        const fetchHumanFormat = async () => {
            try {
                const response = await axios.get(ENDPOINT + "/human_format");
                status_log("Sucessfully loaded human format dictionary!", success)
                return response.data
            }
            catch (error) {
                status_log("Failed to load human format dictionary", failure)
                console.error(error);
                return
            }
        };

        // get the sample instances
        const fetchInstances = async () => {
            try {
                const response = await axios.get(ENDPOINT + "/instances");
                setInstances(response.data);
                setSelectedInstance(response.data[0]);
                status_log("Sucessfully loaded instances!", success)
            } catch (error) {
                status_log("Failed to load instances", failure)
                console.error(error);
            }
        };
        // Call the pageLoad function when the component mounts
        pageLoad();
    }, []);

    // useEffect to handle explanation when the selected instances changes
    useEffect(() => {
        handleExplanations();
    }, [selectedInstance, numExplanations, constraints]);


    useEffect(() => {
        if (explanations.length === 0) return;
        const instanceAndExpls = explanations.map(region => ({
            instance: selectedInstance,
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

    // Function to fetch explanation data from the server
    const handleExplanations = async () => {
        if (constraints.length === 0 || selectedInstance.length == 0) return;

        try {
            status_log("Generated explanation!")
            const response = await axios.post(
                ENDPOINT + "/explanations",
                { selectedInstance, constraints, numExplanations },
            );
            console.log(response.data)
            setExplanations(response.data);
        } catch (error) {
            status_log("Explanation failed", failure)
            console.error(error);
        }
    }

    // Function to handle displaying the previous instances
    const handlePrevApp = () => {
        if (count > 0) {
            setCount(count - 1);
            setSelectedInstance(instances[count - 1]);
        }
        handleExplanations();
    }

    // Function to handle displaying the next instance
    const handleNextApp = () => {
        if (count < instances.length - 1) {
            setCount(count + 1);
            setSelectedInstance(instances[count + 1]);
        }
        handleExplanations();
    }

    const handleNumExplanations = (numExplanations) => () => {
        setNumExplanations(numExplanations);
    }


    // this condition prevents the page from loading until the formatDict is availible
    if (isLoading) {
        return <div></div>
    } else {
        return (
            <div>
                <div>
                    <h2>Application {count}</h2>
                    <button onClick={handlePrevApp}>Previous</button>
                    <button onClick={handleNextApp}>Next</button>

                    {Object.keys(featureDict).map((key, index) => (
                        <div key={index}>
                            <Feature name={formatFeature(key, formatDict)} value={formatValue(selectedInstance[key], key, formatDict)} />
                        </div>
                    ))}
                </div>

                {explanationSection}

            </div>
        )
    }

}


function Feature({ name, value }) {

    return (
        <div className="features-container">
            <div className="feature">
                <p>{name}: <span className="featureValue">{value}</span></p>
            </div>
            {/* Add more similar div elements for each feature */}
        </div>
    )
}


export default App;
