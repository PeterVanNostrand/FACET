import axios, { AxiosError } from 'axios';
import { useEffect, useState } from 'react';
import webappConfig from '../../config.json';
import ExplanationSection from './components/explanations/ExplanationSection.jsx';
import FeatureControlSection from './components/feature-control/FeatureControlSection.jsx';
import ScenarioSection from './components/scenario/ScenarioSection.jsx';
import StatusSection from './components/status/StatusSection.jsx';
import SuggestionSection from './components/suggestion/suggestion.jsx';
import WelcomeScreen from './components/welcome/WelcomeScreen.jsx';
import './css/style.css';

import backArrowIcon from '../icons/BackArrow.svg';

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
     * Structure: [{"x0": <feature value>, ... "x<n>": <feature value>}, {"x0": <feature value>, ... "x<n>": <feature value>}, ...]
     * 
     * selectedInstance: the current application/scenario the user is viewing. This variable is what the app displays
     * Structure: { "x0": <feature value>, ... , "x<n>": <feature value> }
     * 
     * explanation: FACET's counterfactual explanation on what to change user's feature values to to get a desired outcome
     * Structure: Array of size numExplanations objects of regions for each feature
     * [{x0: region, ... , xn: region}, ...]
     * 
     * savedScenarios: List of scenarios user has saved to tabs
     * Structure: [{
     *   "scenario"   : <int>,
     *   "values"     : <selectedInstance>,
     *   "explanation": <explanation>
     * }]
     * 
     * formatDict: JSON instance that contains information regarding formatting and dataset handling
     * Structure: {
            "dataset": <dataset>,
            "feature_decimals": { <featureName>: <val>, ... },
            "feature_names": { "x0": <featureName>, ...},
            "feature_units": { <featureName>: <unit, i.e. "$", "ms", "km">, ... },
            "pretty_feature_names": { <featureName>: <Pretty Feature Name, i.e. "Applicant Income" rather than "ApplicantIncome">, ... },
            "scenario_terms": {
                "desired_outcome": <Val, i.e. "Approved">,
                "instance_name": "<Val, i.e. "Application">,
                "undesired_outcome":<Val, i.e. "Rejected">
            },
            "semantic_max": { <FeatureName>: <Real Number or null>, ... },
            "semantic_min": { <FeatureName>: <Real Number or null>, ... },
            "target_name": <Val, i.e. "Loan_Status">,
            "weight_values": {
                "Increment": <int>,
                "IsExponent": <true or false, determines if we increase weights by the increment or by the power of increment>
            }
        }
     * 
     * featureDict: JSON instance mapping the feature index to a feature name (i.e. x0 -> Apllicant_Income, x1 -> Debt, ...)
     * Structure: { "feature_names": {"x0": <featureName>, ... , "x<n>": <featureName> } }
     * 
     * isLoading: Boolean value that helps with managing async operations. Prevents webapp from trying to display stuff before formatDict is loaded
     */
    const [applications, setApplications] = useState([]);
    const [selectedInstance, setSelectedInstance] = useState("");
    const [selectCustom, setSelectCustom] = useState(null);
    const [customApplicant, setCustomApplicant] = useState(null);
    const [applicantIndex, setApplicantIndex] = useState('0');
    // const [isComparing, setIsComparing] = useState(false); for scenario comparison

    const [featureControls, setFeatureControls] = useState([]);
    const [explanations, setExplanations] = useState("");
    const [numExplanations, setNumExplanations] = useState(10);
    const [totalExplanations, setTotalExplanations] = useState([]);
    const [currentExplanationIndex, setCurrentExplanationIndex] = useState(0);
    const [savedScenarios, setSavedScenarios] = useState([]);
    const [selectedScenarioIndex, setSelectedScenarioIndex] = useState(null);
    const [scenarioCount, setScenarioCount] = useState(1);

    const [formatDict, setFormatDict] = useState(null);
    const [featureDict, setFeatureDict] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const [keepPriority, setKeepPriority] = useState(true);
    const [constraints, setConstraints] = useState([]);
    // const [priorities, setPriorities] = useState(null);

    const [isWelcome, setIsWelcome] = useState(false);


    // fetch instances data when the component mounts
    useEffect(() => {
        status_log("Using endpoint " + ENDPOINT, SUCCESS)

        const pageLoad = async () => {
            fetchInstances();
            let dict_data = await fetchFormatDict()
            setFormatDict(dict_data)
            setFeatureDict(dict_data["feature_names"]);
        }

        // get the human formatting data instances
        const fetchFormatDict = async () => {
            try {
                const response = await axios.get(ENDPOINT + "/data_format");
                status_log("Sucessfully loaded data format dictionary!", SUCCESS);
                return response.data
            }
            catch (error) {
                status_log("Failed to load data format dictionary", FAILURE)
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
        if (selectedScenarioIndex !== null) {
            updateScenario();
        }
    }, [selectedInstance, featureControls, selectedScenarioIndex]);

    useEffect(() => {
        if (explanations.length === 0) {
            setTotalExplanations([]);
            return;
        }

        const instanceAndExpls = explanations.map(region => ({
            instance: selectedInstance,
            region
        }));
        setTotalExplanations(instanceAndExpls);
    }, [explanations])

    // populate features for feature controls
    useEffect(() => {
        if (!formatDict) return;

        try {
            console.log("FORMAT: ", formatDict);
            const priorities = {};
            Object.keys(formatDict.feature_names).forEach((key, index) => {
                priorities[key] = index + 1;
            });
            // setPriorities(priorities);

            const newFeatures = Object.entries(formatDict.feature_names).map(([key, value], index) => {
                // default max when current_value = 0 
                const default_max = 10000;
                // calc. constraints and ranges 
                const current_value = (Math.round(selectedInstance[key]));
                const semantic_min = (formatDict.semantic_min[value] ?? 0);
                const semantic_max = (Math.round(formatDict.semantic_max[value] ?? ((current_value === 0) ? default_max : current_value * 2)));
                const lower_constraint = (Math.round(semantic_max * 0.25));
                const upper_constraint = (Math.round(semantic_max * 0.75));
                setConstraints(prevConstraints => [...prevConstraints, [lower_constraint, upper_constraint]]);
                return {
                    id: value,
                    xid: key,
                    units: formatDict.feature_units[value] || '',
                    title: formatDict.pretty_feature_names[value] || '',
                    current_value: current_value,
                    min: semantic_min,
                    max: semantic_max,
                    priority: priorities[key],
                    lock_state: false,
                    pin_state: false,
                    min_range: lower_constraint,
                    max_range: upper_constraint,
                };
            });

            setFeatureControls(newFeatures);
            setIsLoading(false);
        } catch (error) {
            console.error("Error while populating features:", error);
        }
    }, [formatDict, selectedInstance]);


    /**
     * Function to explain the selected instance using the backend server
     * @returns None
     */
    const handleExplanations = async () => {
        if (constraints.length === 0 || selectedInstance.length == 0) return;

        try {
            const lockOffset = 0.01;
            // build the explanation query, should hold the instance, weights, constraints, etc
            const priorities = {};
            const constraints = {};
            featureControls.forEach((feature, index) => {
                // set priorities
                priorities[feature.xid] = index + 1;
                // set constraints
                if (feature.lock_state) { // with lock
                    const curr_val = parseInt(feature.current_value);
                    constraints[feature.xid] = [curr_val - lockOffset, curr_val + lockOffset]
                }
                else { // without lock
                    constraints[feature.xid] = [parseInt(feature.min_range), parseInt(feature.max_range)];
                }
            });

            // put the min and max ranges in to a list
            const sortedConstraints = Object.fromEntries(Object.entries(constraints).sort());
            const constraintsArray = [];
            for (var xid in sortedConstraints) {
                constraintsArray.push(sortedConstraints[xid]);
            }

            let request = {};
            request["instance"] = selectedInstance;
            request["weights"] = priorities;
            request["constraints"] = constraintsArray;
            request["num_explanations"] = numExplanations;

            // make the explanation request
            const response = await axios.post(
                ENDPOINT + "/explanations",
                request,
            );

            status_log("Successfully generated explanation!", SUCCESS)

            // update the explanation content
            setExplanations(response.data);
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
            scenarioID: scenarioCount,
            instance: selectedInstance,
            features: featureControls,
            explanations: [...explanations],
            explanationIndex: currentExplanationIndex,
            constraints: [...constraints]
        };
        setScenarioCount(scenarioCount + 1);
        setSavedScenarios([...savedScenarios, newScenario]);
    }
    const updateScenario = () => {
        if (selectedScenarioIndex < 0 || selectedScenarioIndex >= savedScenarios.length) {
            // Invalid index
            return;
        }

        const updatedScenario = {
            ...savedScenarios[selectedScenarioIndex],
            features: featureControls,
            explanations: [...explanations],
            explanationIndex: currentExplanationIndex,
            constraints: [...constraints]
        };

        const updatedScenarios = [...savedScenarios];
        updatedScenarios[selectedScenarioIndex] = updatedScenario;
        setSavedScenarios(updatedScenarios);
    }

    // const changeToScenario = () => { // Change to comparing scenarios
    //     setIsComparing(prevIsComparing => !prevIsComparing);
    // };

    const backToWelcomeScreen = () => {
        setIsWelcome(true);
    }

    if (isLoading) {
        return <></>
    }
    else if (isWelcome) {
        return (
            <WelcomeScreen
                instances={applications}
                selectedInstance={selectedInstance}
                setSelectedInstance={setSelectedInstance}
                setIsWelcome={setIsWelcome}
                formatDict={formatDict}
                featureDict={featureDict}
                selectCustom={selectCustom}
                setSelectCustom={setSelectCustom}
                customApplicant={customApplicant}
                setCustomApplicant={setCustomApplicant}
                setApplicantIndex={setApplicantIndex}
                applicantIndex={applicantIndex}
            />
        )
    } else {
        return (
            <div id="super-div" className="super-div">
                <div id="back-welcome-grid" className="card">
                    <button className="back-welcome-button" onClick={backToWelcomeScreen}>
                        <img src={backArrowIcon} alt="Back Arrow Icon" />
                        {formatDict["scenario_terms"]["instance_name"]}
                    </button>
                    <h1 id="app-logo">
                        FACET
                    </h1>
                    {/* <button
                        onClick={changeToScenario}>
                            Experimental Comparison Mode
                    </button> */}
                </div>

                <FeatureControlSection
                    features={featureControls}
                    setFeatures={setFeatureControls}
                    constraints={constraints}
                    setConstraints={setConstraints}
                    keepPriority={keepPriority}
                    setKeepPriority={setKeepPriority}
                    savedScenarios={savedScenarios}
                    selectedScenarioIndex={selectedScenarioIndex}
                    setSelectedScenarioIndex={setSelectedScenarioIndex}
                    formatDict={formatDict}
                />

                <ScenarioSection
                    savedScenarios={savedScenarios}
                    setSavedScenarios={setSavedScenarios}
                    setExplanations={setExplanations}
                    setCurrentExplanationIndex={setCurrentExplanationIndex}
                    setSelectedInstance={setSelectedInstance}
                    selectedScenarioIndex={selectedScenarioIndex}
                    setSelectedScenarioIndex={setSelectedScenarioIndex}
                    setConstraints={setConstraints}
                    setScenarioCount={setScenarioCount}
                />

                <div id="status-grid" className="card">
                    <StatusSection
                        instance={selectedInstance}
                        featureDict={featureDict}
                        formatDict={formatDict}
                    />
                </div>
                {/* {isComparing ? (
                    <div id="scenario-comparsion-grid" className="card">
                        <ScenarioComparison savedScenarios={savedScenarios} />
                    </div>
                ) : (  */}
                <div id="explanation-grid" className={`card`}>
                    <ExplanationSection
                        key={currentExplanationIndex}
                        explanations={explanations}
                        totalExplanations={totalExplanations}
                        formatDict={formatDict}
                        currentExplanationIndex={currentExplanationIndex}
                        setCurrentExplanationIndex={setCurrentExplanationIndex}
                        saveScenario={saveScenario}
                    />
                </div>
                {/* )}  */}
                <div id="suggestion-grid" className="card">
                    <SuggestionSection
                        formatDict={formatDict}
                        selectedInstance={selectedInstance}
                        explanations={explanations}
                        currentExplanationIndex={currentExplanationIndex}
                        featureDict={featureDict} />
                </div>
            </div >
        )
    }

}

export default App;