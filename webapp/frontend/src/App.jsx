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
    const [savedInstanceList, setSavedInstanceList] = useState([]);
    const [masterObj, setMasterObj] = useState({});
    const [instances, setInstances] = useState([]);
    const [index, setIndex] = useState(0);
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
            setMasterObj(dict_data);
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
                if(instances.length > 0){
                    status_log("Sucessfully loaded instances!", SUCCESS)
                }
                else{
                    status_log("Failed to load instances (length 0)", FAILURE);
                }
            } catch (error) {
                status_log("Failed to load instances", FAILURE)
                console.error(error);
            }
        };
        // Call the pageLoad function when the component mounts
        pageLoad();
    }, [instances]);

    //fetches the weights of the features
    const getWeights = () => {
        let weights = {};
        for (let feature in masterObj["feature_names"]) {
            let priority = masterObj["feature_names"][feature]["currPriority"];
            let w = 1; //the weight for this feature
            if (masterObj["weight_values"]["IsExponent"]) {
                //if feature is locked, w = 1; else increment the weight appropriately
                w = masterObj["feature_names"][feature]["locked"] ? 1 : Math.pow(priority, masterObj["weight_values"]["Increment"]);
            }
            else {
                //if feature is locked, w = 1; else increment the weight appropriately
                w = masterObj["feature_names"][feature]["locked"] ? 1 : (1 + (priority - 1) * masterObj["weight_values"]["Increment"]);
            }
            weights[feature] = w;
        }
        return (weights);
    }

    // useEffect to handle explanation when the selected instances changes
    useEffect(() => {
        handleExplanation();
    }, [index]);

    /**
     * Function to explain the selected instance using the backend server
     * @returns None
     */
    const handleExplanation = async () => {
        try {
            // if we have no instance to explain, there's nothing to do
            if (instances[index] == "")
                return;
            // build the explanation query, should hold the instance, weights, constraints, etc
            let query_data = {};
            query_data["instance"] = instances[index]
            query_data["weights"] = getWeights();
            // console.debug("query data is:")
            // console.debug(query_data)
            status_log("query data is:", DEBUG)
            console.debug(query_data)


            // make the explanation request
            const response = await axios.post(
                ENDPOINT + "/explanation",
                query_data,
            );
            // update the explanation content
            instances[index]["explanation"] = response.data;
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
        if (index > 0) {
            setIndex(index - 1);
        }
        else{ //cycle back to the top
            setIndex(instances.length - 1);
        }
        handleExplanation();
    }

    // Function to handle displaying the next instance
    const handleNextApp = () => {
        if (index < instances.length - 1) {
            setIndex(index + 1);
        }
        else{ //cycle back to beginning
            setIndex(0);
        }
        handleExplanation();
    }


    // this condition prevents the page from loading until the formatDict is availible
    if (isLoading) {
        return <div></div>
    } else {
        return (
            <>
                <div>
                    <h2>Application {index}</h2>
                    <button onClick={handlePrevApp}>Previous</button>
                    <button onClick={handleNextApp}>Next</button>

                    {Object.keys(masterObj["feature_names"]).map((key, index) => (
                        <div key={index}>
                            <Feature name={masterObj["pretty_feature_names"][masterObj["feature_names"][key]]} value={formatValue(instances[index][key], key, masterObj)} />
                        </div>
                    ))}
                </div>

                <h2>Explanation</h2>


                {Object.keys(instances[index]["explanation"]).map((key, index) => (
                    <div key={index}>
                        <h3>{masterObj["pretty_feature_names"][masterObj["feature_names"][key]]}</h3>
                        <p>{formatValue(instances[index]["explanation"][key][0], key, masterObj)}, {formatValue(instances[index]["explanation"][key][1], key, masterObj)}</p>
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
