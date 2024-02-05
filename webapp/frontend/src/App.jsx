import axios, { AxiosError } from 'axios';
import { useEffect, useState } from 'react';
import webappConfig from '../../config.json';
import StatusDisplay from './StatusDisplay.jsx';
import WelcomeScreen from './WelcomeSceen.jsx';
import './css/style.css';
import './css/feature-control.css'

import FeatureControlSection from './FeatureControlSection.jsx';
import ExplanationSection from './components/my-application/explanation/ExplanationSection';

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
    /**
     * applications: List of applications loaded from JSON data
     * Structure:
     * [{"x0": <feature value>, ... "x<n>": <feature value>}, {"x0": <feature value>, ... "x<n>": <feature value>}, ...]
     * 
     * index: index of where we are in applications; an int in range [0, applications.length - 1]
     * 
     * selectedInstance: the current application/scenario the user is viewing. This variable is what the app displays
     * Structure:
     * {
        "x0": <feature value>,
                ⋮
        "x<n>": <feature value>
       }
     * 
     * explanation: FACET's counterfactual explanation on what to change user's feature values to to get a desired outcome
     * Structure:
     * {
        "x0": [
            <Region Min (float)>,
            <Region Max (float)>
        ],
                    ⋮
        "x<n>": [
            <Region Min (float),
            <Region Max (float)
        ]
        }
     *}
     * 
     * savedScenarios: List of scenarios user has saved to tabs
     * Structure:
     * [{
     *   "scenario"   : <int>,
     *   "values"     : <selectedInstance>,
     *   "explanation": <explanation>
     * }]
     * 
     * formatDict: JSON instance that contains information regarding formatting and dataset handling
     * Structure:
     * {
            "dataset": <dataset>,
            "feature_decimals": {
                <featureName>: <val>,
                <featureName>: <val>,
                        ⋮
                <featureName>:<val>
            },
            "feature_names": {
                "x0": <featureName>.
                        ⋮
                "x<n>":<featureName>
            },
            "feature_units": {
                <featureName>: <unit, i.e. "$", "ms", "km">,
                        ⋮
                <featureName>:<unit>
            },
            "pretty_feature_names": {
                <featureName>: <Pretty Feature Name, i.e. "Applicant Income" rather than "ApplicantIncome">,
                            ⋮
                <featureName>:<Pretty Feature Name>
            },
            "scenario_terms": {
                "desired_outcome": <Val, i.e. "Approved">,
                "instance_name": "<Val, i.e. "Application">,
                "undesired_outcome":<Val, i.e. "Rejected">
            },
            "semantic_max": {
                <FeatureName>: <Real Number or null>,
                                ⋮
                <FeatureName>:<Real Number or null>
            },
            "semantic_min": {
                <FeatureName>: <Real Number or null>,
                                ⋮
                <FeatureName>:<Real Number or null>
            },
            "target_name": <Val, i.e. "Loan_Status">,
            "weight_values": {
                "Increment": <int>,
                "IsExponent": <true or false, determines if we increase weights by the increment or by the power of increment>
            }
        }
     * 
     * featureDict: JSON instance mapping the feature index to a feature name (i.e. x0 -> Apllicant_Income, x1 -> Debt, ...)
     * Structure:
     * {
            "feature_names": {
                "x0": <featureName>.
                        ⋮
                "x<n>":<featureName>
            },
        }
     * 
     * isLoading: Boolean value that helps with managing async operations. Prevents webapp from trying to display stuff before formatDict is loaded
     */
    const [applications, setApplications] = useState([]);
    // const [index, setIndex] = useState(0);
    const [selectedInstance, setSelectedInstance] = useState("");
    const [explanations, setExplanations] = useState("");
    const [savedScenarios, setSavedScenarios] = useState([]);
    const [formatDict, setFormatDict] = useState(null);
    const [featureDict, setFeatureDict] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const [constraints, setConstraints] = useState([]);
    const [numExplanations, setNumExplanations] = useState(1);
    const [totalExplanations, setTotalExplanations] = useState([]);
    const [explanationSection, setExplanationSection] = useState(null);
    const [isWelcome, setIsWelcome] = useState(false);
    const [showWelcomeScreen, setShowWelcomeScreen] = useState(false);

    console.log('app constraints', constraints)

    useEffect(() => {
        console.log('constraints', constraints)
    }, [constraints])

    // useEffect to fetch instances data when the component mounts
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
                setApplications(response.data);
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
        return weights;
    }

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


    const handleNumExplanations = (numExplanations) => () => {
        setNumExplanations(numExplanations);
    }

    const backToWelcomeScreen = () => {
        setShowWelcomeScreen(true);
    }

    const welcome = WelcomeScreen(showWelcomeScreen, setShowWelcomeScreen, selectedInstance, setSelectedInstance)

    /**
     * Saves a scenario to savedScenarios, and creates a tab
     */
    const saveScenario = () => {
        let scenario = {}; //made this way for programmer's convenience
        scenario["scenario"] = savedScenarios.length + 1; //ID the scenario indexing at 1
        scenario["values"] = selectedInstance; //store feature values
        scenario["explanation"] = explanations; //store explanation
        scenario["featureControls"] = {}; //TODO: store priorities of features, lock states, etc.

        setSavedScenarios([...savedScenarios, scenario]); //append scenario to savedScenarios        
        //Create new tab and add it to HTML
        let tab = document.createElement("button");
        tab.innerHTML = "Scenario " + (savedScenarios.length + 1); //Name the tab
        //set onclick method to load the scenario, and display the ID
        tab.onclick = function () { setSelectedInstance(scenario["values"]), document.getElementById("title").innerHTML = "Scenario " + scenario["scenario"] };
        document.getElementById("tabSection").appendChild(tab); //add element to HTML
    }


    // useEffect(() => {
    //     if (formatDict && featureDict && showWelcomeScreen) {
    //         setWelcomeScreenContent(
    //             <WelcomeScreen
    //                 applications={applications}
    //                 handleApplicationChange={handleApplicationChange}
    //                 selectedInstance={selectedInstance}
    //                 featureDict={{ ...featureDict }}
    //                 formatDict={{ ...formatDict }}
    //                 setSelectedInstance={{ ...setSelectedInstance }}
    //                 index={index}
    //                 setShowWelcomeScreen={setShowWelcomeScreen}
    //             />
    //         )
    //     }
    // }, [formatDict, featureDict, showWelcomeScreen])


    // this condition prevents the page from loading until the formatDict is availible
    if (isLoading) {
        return <div></div>
    }
    else if (showWelcomeScreen) {
        let welcomeContent = welcome

        if (welcomeContent["status"] == "Display") {
            return welcomeContent["content"]
        } else {
            console.log("The content changed!")
            setShowWelcomeScreen(false)

            if (welcomeContent["content"] != null) {
                setSelectedInstance(welcomeContent["content"])
            }
        }
    } else {
        return (
            <div id="super-div" className="super-div">
                <div id="settings-profile-section" className="settings-profile-section">
                    <button onClick={backToWelcomeScreen}>← Welcome Screen</button>
                </div>

                <div id="feature-controls" className="card feature-controls">
                    <FeatureControlSection
                        applicantInfo={selectedInstance}
                        fDict={formatDict}
                        constraints={constraints}
                        setConstraints={setConstraints}
                    />
                </div>

                <div id="tab-section" className="tab-section">
                    <h2>Tabs</h2>
                    <div id="tabSection" style={{
                        display: "flex",
                        flexDirection: "row",
                    }}>
                    </div>
                </div>

                <div id="status-section" className="card status-section">
                    <h2 id="title">Application</h2>
                    <StatusDisplay featureDict={featureDict} formatDict={formatDict} selectedInstance={selectedInstance} />
                </div>

                <div id="explanation" className="card explanation">
                    <h2 className='explanation-header' style={{}}>
                        Explanation
                    </h2>
                    {explanationSection}
                    <button onClick={saveScenario}>Save Scenario</button>
                </div>

                <div id="suggestion" className="card suggestion">
                    <p>suggestions box thing</p>
                </div>
            </div>
        )
    }

}

export default App;