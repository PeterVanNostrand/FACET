import axios from "axios";
import { useEffect, useState } from "react";
import webappConfig from "../../config.json";
import { formatFeature, formatValue } from "../utilities";

const success = "Lime"
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
        status_log("Using endpoint " + ENDPOINT, success)

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
        handleExplanation();
    }, [selectedInstance]);


    // Function to fetch explanation data from the server
    const handleExplanation = async () => {
        try {
            status_log("Generated explanation!")
            const response = await axios.post(
                ENDPOINT + "/explanation",
                selectedInstance,
            );
            setExplanation(response.data);
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
