import axios, { AxiosError } from "axios";
import { useEffect, useState } from "react";
import webappConfig from "../../config.json";
import ExplanationDisplay from "./ExplanationDisplay.jsx";
import StatusDisplay from './StatusDisplay.jsx';
import "./css/style.css";

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
    const [explanation, setExplanation] = useState("");
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
            console.debug(query_data)

            // make the explanation request
            const response = await axios.post(
                ENDPOINT + "/explanation",
                query_data,
            );
            // update the explanation content
            setExplanation(response.data);
            status_log("Successfully generated explanation!", SUCCESS)
            console.debug(response.data)

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
        if (count > 0) {
            setCount(count - 1);
            setSelectedInstance(instances[count - 1]);
        }
        handleExplanation();
    }

    // Function to handle displaying the next instance
    const handleNextApp = () => {
        if (count < instances.length - 1) {
            setCount(count + 1);
            setSelectedInstance(instances[count + 1]);
        }
        handleExplanation();
    }


    // this condition prevents the page from loading until the formatDict is availible
    if (isLoading) {
        return <div></div>
    } else {
        return (
            <>
                <div id="super-div" className="super-div">
                    <div id="settings-profile-section" className="settings-profile-section">
                        <p>settings profile stuff</p>
                    </div>
                    <div id="tab-section" className="tab-section">
                        <p>tabs</p>
                    </div>
                    <div id="feature-controls" className="feature-controls">
                        <p>feature  controls</p>
                    </div>
                    <div id="status-section" className="status-section">
                        <h2>Application {count}</h2>
                        <button onClick={handlePrevApp}>Previous</button>
                        <button onClick={handleNextApp}>Next</button>
                        <StatusDisplay featureDict={featureDict} formatDict={formatDict} selectedInstance={selectedInstance} />
                    </div>
                    <div id="explanation" className="explanation">
                        <h2>Explanation</h2>
                        <ExplanationDisplay explanation={explanation} formatDict={formatDict} />
                    </div>
                    <div id="suggestion" className="suggestion">
                        <p>suggestions box thing</p>
                    </div>
                </div>
            </>
        )
    }

}

export default App;
