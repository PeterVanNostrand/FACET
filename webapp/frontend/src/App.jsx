import axios, { AxiosError } from 'axios';
import { useEffect, useState } from 'react';
import webappConfig from '../../config.json';
import StatusDisplay from './components/StatusDisplay.jsx';
import WelcomeScreen from './components/welcome/WelcomeSceen.jsx';
import './css/style.css';

import ScenarioSection from './components/ScenarioSection.jsx';
import ExplanationSection from './components/explanations/ExplanationSection';
import FeatureControlSection from './components/feature-control/FeatureControlSection.jsx';

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
    const [selectedInstance, setSelectedInstance] = useState("");

    const [features, setFeatures] = useState([]);
    const [explanations, setExplanations] = useState("");
    const [numExplanations, setNumExplanations] = useState(10);
    const [totalExplanations, setTotalExplanations] = useState([]);
    const [currentExplanationIndex, setCurrentExplanationIndex] = useState(0);

    const [savedScenarios, setSavedScenarios] = useState([]);
    const [formatDict, setFormatDict] = useState(null);
    const [featureDict, setFeatureDict] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const [constraints, setConstraints] = useState([
        [1000, 1600], [0, 10], [6000, 10000], [300, 500]
    ]);

    const [isWelcome, setIsWelcome] = useState(false);
    const [showWelcomeScreen, setShowWelcomeScreen] = useState(false);
    const [keepPriority, setKeepPriority] = useState(true);

    // useEffect to fetch instances data when the component mounts
    useEffect(() => {
        status_log("Using endpoint " + ENDPOINT, SUCCESS)

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
        pageLoad();
    }, []);


    useEffect(() => {
        handleExplanations();
        console.log('updated constraintsw')
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
        try {
            if (formatDict) {
                let priorityValue = 1;

                const newFeatures = Object.entries(formatDict.feature_names).map(([key, value], index) => {
                    const currentValue = selectedInstance[key];
                    const isZero = currentValue === 0; // checks if current feature value = zero

                    const default_max = 1000;
                    const default_max_range = 500;
                    const default_min_range = 0;

                    const lowerConstraint = constraints[index][0]
                    const upperConstraint = constraints[index][1]

                    return {
                        id: value,
                        x: key,
                        units: formatDict.feature_units[value] || '',
                        title: formatDict.pretty_feature_names[value] || '',
                        current_value: currentValue,
                        min: formatDict.semantic_min[value] ?? 0,
                        max: formatDict.semantic_max[value] ?? (isZero ? default_max : currentValue * 2), // set 1 if null or double current_val if income is not 0
                        priority: priorityValue++,
                        lock_state: false,
                        pin_state: false,
                        min_range: lowerConstraint,
                        max_range: upperConstraint,
                    };
                });

                setFeatures(newFeatures);
                console.log("features: ", features);
            }
        } catch (error) {
            console.error("Error while populating features:", error);
        }
    }, [formatDict]);

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

    const saveScenario = () => {
        const newScenario = {
            scenarioID: savedScenarios.length + 1,
            instance: selectedInstance,
            explanationIndex: currentExplanationIndex,
            constraints: [...constraints]
        };

        setSavedScenarios([...savedScenarios, newScenario]);
    }


    const backToWelcomeScreen = () => {
        setShowWelcomeScreen(true);
    }

    const welcome = WelcomeScreen(showWelcomeScreen, setShowWelcomeScreen, selectedInstance, setSelectedInstance)

    if (isLoading) {
        return <></>
    }
    else if (showWelcomeScreen) {
        let welcomeContent = welcome

        if (welcomeContent["status"] == "Display") {
            return welcomeContent["content"]
        } else {
            setShowWelcomeScreen(false);

            if (welcomeContent["content"] != null) {
                setSelectedInstance(welcomeContent["content"])
            }
        }
    } else {
        return (
            <div id="super-div" className="super-div">
                <div id="back-welcome-grid" className="card">
                    <button className="back-welcome-button" onClick={backToWelcomeScreen}>← Welcome Screen</button>
                    <h1 id="app-logo">
                        FACET
                    </h1>
                </div>

                <div id="scenario-grid" className="card">
                    <ScenarioSection
                        savedScenarios={savedScenarios}
                        setSavedScenarios={setSavedScenarios}
                        setCurrentExplanationIndex={setCurrentExplanationIndex}
                        setSelectedInstance={setSelectedInstance}
                        setConstraints={setConstraints}
                    />
                </div>

                <div id="feature-controls-grid" className="card">
                    <FeatureControlSection
                        features={features}
                        setFeatures={setFeatures}
                        constraints={constraints}
                        setConstraints={setConstraints}
                        keepPriority={keepPriority}
                        setKeepPriority={setKeepPriority}
                    />
                </div>

                <div id="status-grid" className="card">
                    <h2>My Application</h2>
                    <StatusDisplay
                        featureDict={featureDict}
                        formatDict={formatDict}
                        selectedInstance={selectedInstance}
                    />
                </div>

                <div id="explanation-grid" className="card">
                    {totalExplanations.length > 0 &&
                        <ExplanationSection
                            explanations={explanations}
                            totalExplanations={totalExplanations}
                            formatDict={formatDict}
                            currentExplanationIndex={currentExplanationIndex}
                            setCurrentExplanationIndex={setCurrentExplanationIndex}
                        />
                    }
                    <button onClick={saveScenario}>Save Scenario</button>
                </div>

                <div id="suggestion-grid" className="card">
                    <p>suggestions</p>
                </div>
            </div>
        )
    }

}

export default App;