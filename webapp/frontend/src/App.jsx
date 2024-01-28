import axios, { AxiosError } from "axios";
import { useEffect, useState } from "react";
import webappConfig from "../../config.json";
import { formatFeature, formatValue } from "../utilities";
import "./css/App.css";

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
    const [index, setIndex] = useState(0);
    const [selectedInstance, setSelectedInstance] = useState("");
    const [explanation, setExplanation] = useState("");
    const [savedScenarios, setSavedScenarios] = useState([]);
    const [formatDict, setFormatDict] = useState(null);
    const [featureDict, setFeatureDict] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    // determine the server path
    const SERVER_URL = webappConfig.SERVER_URL
    const API_PORT = webappConfig.API_PORT
    const ENDPOINT = SERVER_URL + ":" + API_PORT + "/facet"

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

        // get the hman formatting data instances
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

    // useEffect to handle explanation when the selected instances changes
    useEffect(() => {
        handleExplanation();
    }, [selectedInstance]);

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
    /**
     * Function to explain the selected instance using the backend server
     * @returns None
     */
    const handleExplanation = async () => {
        try {
            // if we have no instance to explain, there's nothing to do
            if (selectedInstance == "")
                return;
            // build the explanation query, should hold the instance, weights, constraints, etc
            let query_data = {};
            query_data["instance"] = selectedInstance
            query_data["weights"] = getWeights();
            status_log("query data is:", DEBUG)

            // make the explanation request
            const response = await axios.post(
                ENDPOINT + "/explanation",
                query_data,
            );
            // update the explanation content
            setExplanation(response.data);
            status_log("Successfully generated explanation!", SUCCESS)

        }
        catch (error) {
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
        if (index > 0) {
            setIndex(index - 1);
            document.getElementById("title").innerHTML = "Application " + (index - 1); //Need to change back to application if a saved scenario was visited
            setSelectedInstance(applications[index - 1]);
        }
        else{ //Cycle back to last application in instance list
            setIndex(applications.length -1);
            document.getElementById("title").innerHTML = "Application " + (applications.length - 1); //Need to change back to application if a saved scenario was visited
            setSelectedInstance(applications[applications.length - 1]);
        }
        handleExplanation();
    }

    // Function to handle displaying the next instance
    const handleNextApp = () => {
        if (index < applications.length - 1) {
            setIndex(index + 1);
            document.getElementById("title").innerHTML = "Application " + (index + 1); //Need to change back to application if a saved scenario was visited
            setSelectedInstance(applications[index + 1]);
        }
        else{ //cycle back to first application in instance list
            setIndex(0);
            setSelectedInstance(applications[0]);
            document.getElementById("title").innerHTML = "Application 0"; //Need to change back to application if a saved scenario was visited
        }
        handleExplanation();
    }

    /**
     * Saves a scenario to savedScenarios, and creates a tab
     */
    const saveScenario = () => {
        let scenario = {}; //made this way for programmer's convenience
        scenario["scenario"] = savedScenarios.length + 1; //ID the scenario indexing at 1
        scenario["values"] = selectedInstance; //store feature values
        scenario["explanation"] = explanation; //store explanation
        scenario["featureControls"] = {}; //TODO: store priorities of features, lock states, etc.
        
        setSavedScenarios([...savedScenarios, scenario]); //append scenario to savedScenarios        
        //Create new tab and add it to HTML
        let tab = document.createElement("button");
        tab.innerHTML = "Scenario " + (savedScenarios.length + 1); //Name the tab
        //set onclick method to load the scenario, and display the ID
        tab.onclick = function() {setSelectedInstance(scenario["values"]), document.getElementById("title").innerHTML = "Scenario " + scenario["scenario"]};
        document.getElementById("tabSection").appendChild(tab); //add element to HTML
    }


    // this condition prevents the page from loading until the formatDict is availible
    if (isLoading) {
        return <div></div>
    } else {
        return (
            <>
                <div>
                    <div id="tabSection" style={{
                        display:"flex", 
                        flexDirection:"row",
                        }}>
                    </div>
                    <h2 id="title">Application {index}</h2>
                    <button onClick={handlePrevApp}>Previous</button>
                    <button onClick={handleNextApp}>Next</button>
                    <button onClick={saveScenario}>Save Scenario</button>

                    {Object.keys(featureDict).map((key, index) => (
                        <div key={index}>
                            <Feature name={formatFeature(key, formatDict)} value={formatValue(selectedInstance[key], key, formatDict)} />
                        </div>
                    ))}
                </div>

                <h2>Explanation</h2>


                {Object.keys(explanation).map((key, index) => (
                    <div key={index}>
                        <h3>{formatFeature(key, formatDict)}</h3>
                        <p>{formatValue(explanation[key][0], key, formatDict)}, {formatValue(explanation[key][1], key, formatDict)}</p>
                    </div>
                ))}
            </>
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