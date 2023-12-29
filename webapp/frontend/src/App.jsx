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

    const handleApplicationChange = (event) => {
        setCount(event.target.value);
        setSelectedInstance(instances[event.target.value]);
    };

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


    // this condition prevents the page from loading until the formatDict is availible
    if (isLoading) {
        return <div></div>
    } else {
        return (
            <div className='main-container' style={{ maxHeight: '98vh', }}>

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
                                {instances.map((applicant, index) => (
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
                                {Object.keys(selectedInstance).map((key, index) => (
                                    <div key={index} className='feature' style={{ margin: -10 }}>
                                        <Feature
                                            name={featureDict[key]}
                                            constraint={constraints[index]}
                                            value={selectedInstance[key]}
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

}


function Feature({ name, constraint, value, updateConstraint }) {
    return (
        <div>
            <strong>{name}</strong>&nbsp;
            : <span className="featureValue">{value}</span>
        </div>
    )
}


export default App;
