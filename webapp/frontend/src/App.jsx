import axios, { AxiosError } from "axios";
import { useEffect, useState } from "react";
import webappConfig from "../../config.json";
import { formatFeature, formatValue } from "../utilities";

import ExplanationSection from './components/my-application/explanation/ExplanationSection';
import NavBar from './components/NavBar';
import FeatureControlSection from './components/feature-control/FeatureControlSection';
import TempWelcome from './TempWelcome.jsx'

import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import { autoType, select } from "d3";

const SERVER_URL = webappConfig.SERVER_URL
const API_PORT = webappConfig.API_PORT
const ENDPOINT = SERVER_URL + ":" + API_PORT + "/facet"

const SUCCESS = "Lime"
const FAILURE = "Red"
const DEBUG = "White"

/**
 * A simple function for neat logging
 * @param {string} text     The status message text
 * @param {string} color    The CSS color to display the message in, affects console output
 */
function status_log(text, color) {
    if (color === null) {
        console.log(text)
    }
    else if (color == FAILURE) {
        console.error("%c" + text, "color:" + color + ";font-weight:bold;");
    }
    else if (color == DEBUG) {
        console.debug("%c" + text, "color:" + color + ";font-weight:bold;");
    }
    else {
        console.log("%c" + text, "color:" + color + ";font-weight:bold;");
    }
}

function App() {
    const [instances, setInstances] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedInstance, setSelectedInstance] = useState("");
    const [explanations, setExplanations] = useState("");
    const [formatDict, setFormatDict] = useState(null);
    const [featureDict, setFeatureDict] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [constraints, setConstraints] = useState([]);
    const [numExplanations, setNumExplanations] = useState(1);
    const [totalExplanations, setTotalExplanations] = useState([]);
    const [explanationSection, setExplanationSection] = useState(null);
    const [isWelcome, setIsWelcome] = useState(false);

    // initialize the page
    useEffect(() => {
        status_log("Using endpoint " + ENDPOINT, SUCCESS)

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

        // get the human formatting data instances
        const fetchHumanFormat = async () => {
            try {
                const response = await axios.get(ENDPOINT + "/human_format");
                status_log("Sucessfully loaded human format dictionary!", SUCCESS)
                return response.data
            }
            catch (error) {
                status_log("Failed to load human format dictionary", FAILURE)
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
                status_log("Sucessfully loaded instances!", SUCCESS)
            } catch (error) {
                status_log("Failed to load instances", FAILURE)
                console.error(error);
            }
        };
        // Call the pageLoad function when the component mounts
        pageLoad();
    }, []);

    useEffect(() => {
        handleExplanations();
    }, [selectedInstance, numExplanations, constraints]);

    //fetches the weights of the features
    const getWeights = () => {
        let weights = {};
        for (let feature in featureDict) {
            let priority = featureDict[feature]["currPriority"];
            let w = 1; //the weight for this feature
            if (formatDict["weight_values"]["IsExponent"]) {
                //if feature is locked, w = 1; else increment the weight appropriately
                w = featureDict[feature]["locked"] ? 1 : Math.pow(priority, formatDict["weight_values"]["Increment"]);
            }
            else {
                //if feature is locked, w = 1; else increment the weight appropriately
                w = featureDict[feature]["locked"] ? 1 : (1 + (priority - 1) * formatDict["weight_values"]["Increment"]);
            }
            weights[feature] = w;
        }
        return (weights);
    }

    useEffect(() => {
        if (explanations.length === 0) return;

        console.log('expl', explanations)
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


    /**
     * Function to explain the selected instance using the backend server
     * @returns None
     */
    const handleExplanations = async () => {
        if (constraints.length === 0 || selectedInstance.length == 0) return;

        try {
            // build the explanation query, should hold the instance, weights, constraints, etc
            let request = {};
            request["instance"] = selectedInstance
            request["weights"] = getWeights();
            request["constraints"] = constraints;
            request["num_explanations"] = numExplanations;

            // make the explanation request
            const response = await axios.post(
                ENDPOINT + "/explanations",
                request,
            );

            // update the explanation content
            setExplanations(response.data);
            status_log("Successfully generated explanation!", SUCCESS)
            console.debug(response.data)
        } catch (error) {
            if (error.code != AxiosError.ECONNABORTED) { // If the error is not a front end reload
                let error_text = "Failed to generate explanation (" + error.code + ")"
                error_text += "\n" + error.message
                if (error.response) {
                    error_text += "\n" + error.response.data;
                }
                status_log(error_text, FAILURE)
            }
            return;
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
    } else if (isWelcome) {
        return <TempWelcome
            setIsWelcome={setIsWelcome}
            instances={instances}
            handleApplicationChange={handleApplicationChange}
            count={count}
        />
    } else {
        return (
            <div className='main-container' style={{ maxHeight: '98vh', }}>
                <button
                    onClick={() => setIsWelcome(true)}
                >
                    Back to Welcome Page
                </button>
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
