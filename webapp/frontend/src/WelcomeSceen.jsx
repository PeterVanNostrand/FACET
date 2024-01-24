import React, { useEffect, useState, useRef } from 'react';
import { visualDisplay } from '../../../visualization/src/visualDisplay.js';
import * as d3 from 'd3'
import { json } from 'd3';
import './css/welcomescreen.css'
import InformationSVG from './SVG/Information.svg'
import CloseSVG from './SVG/XClose.svg'
import webappConfig from "../../config.json";
import axios from "axios";
import { formatFeature, formatValue } from "../utilities";
import { SelectionHelpers } from 'victory';
import Dropdown from 'react-dropdown';
import 'react-dropdown/style.css';

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

const WelcomeScreen = ({ applicationList, scenarioList }) => {
    const [currentTab, setCurrentTab] = useState(0)
    const [selected, setsSelected] = useState(false)
    const [dropdownvalue, setDropdownvalue] = useState(null)

    const [instances, setInstances] = useState([]);
    const [count, setCount] = useState(0);
    const [selectedInstance, setSelectedInstance] = useState("");
    const [explanation, setExplanation] = useState("");
    const [formatDict, setFormatDict] = useState(null);
    const [featureDict, setFeatureDict] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const SERVER_URL = webappConfig.SERVER_URL
    const API_PORT = webappConfig.API_PORT
    const ENDPOINT = SERVER_URL + ":" + API_PORT + "/facet"


    let applications = new Map();
    let dropDownApplicaitons = [];


    //Kepe the Selected Application between tabs
    //Add a close button that doesn't change the state
    //randy-b.3-constarints -> App Line 99

    // Loading the need instances for the component
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

            return true
        };

        // Call the pageLoad function when the component mounts
        pageLoad();
    }, []);

    const validTabNumber = (number) => {
        // TODO: Code that checks to see whether the given tab 
        //number is valid and has a corresponding tab

        if (number == 0 || number == 1) {
            return true
        } else {
            return false
        }
    }

    // If the user clicks on a tab, 
    // - Make sure that they clicked on the tab they aren't current one
    // - Check to make sure that the number has a corresponding tab
    // - Set the Current Tab to the new tab and update
    const handleTabControl = (number) => {
        if (currentTab != number && validTabNumber(number)) {
            setCurrentTab(number)
            //setSelectedInstance(instances[0]);
            //setDropdownvalue(dropDownApplicaitons[0])
        }
    }

    const handleInformationClick = () => {
        console.log("Information Now!");
    }

    const handleCloseClick = () => {
        setsSelected(true) //Have it pull from previous applicant before this tab was open
    }

    const handleConfirmButton = () => {
        // TODO: Pass either current applicant or custom applicant and scenario (from tab number)
        // and start the visualization

        //If cusstom applicant, make sure all field are filled with valid inputs before continuing 
        let finalizedInstance = ""

        if (currentTab == 1){
            //If on custom applicant tab

            let customInstance = {}

            let names = Object.getOwnPropertyNames(featureDict)

            for (let prop in names){
                let featureName = names[prop]
                console.log("FeatureInput" + featureName + " " + document.getElementById("FeatureInput" + featureName).textContent)
                customInstance[featureName] = document.getElementById("FeatureInput" + featureName).textContent   
            }

            console.log(customInstance)

            finalizedInstance = customInstance

        } else {
            finalizedInstance = selectedInstance
        }

        setSelectedInstance(finalizedInstance)

        setsSelected(true);
    }

    const handleDropDownChange = (value) => {
        setDropdownvalue(value)
        setSelectedInstance(applications.get(value))
    }

    const getDetailedFeaturesOfSelected = () => {
        return <div>{Object.keys(featureDict).map((key, index) => (
            <div key={index}>
                <Feature name={formatFeature(key, formatDict)} value={formatValue(selectedInstance[key], key, formatDict)} />
            </div>
        ))}</div>
    }

    const FeatureInputs = () => {
        return <div>{Object.keys(featureDict).map((key, index) => (
            <div key={index}>
                <FeatureInputTest 
                prettyName={formatFeature(key, formatDict)} 
                name={key}
                updateValue={(newValue) => (console.log(name + " has been changed to the value " + newValue))} />
            </div>
        ))}</div>

    }

    for (let i = 0; i < instances.length; i++) {
        applications.set("Application " + (i + 1), instances[i]);
        dropDownApplicaitons.push("Application " + (i + 1));
    }

    if (isLoading) {
        // If the files are still loading
        return <div></div>
    } else if (selected) {
        // If a valid application has been selected
        return getDetailedFeaturesOfSelected();

    } else {
        console.log(featureDict)
        console.log(selectedInstance)
        //If an application is still being selected

        //Suppose to set this, however, it doesn't matter since SelectedInstances changes with DropDownValue anyways
        //setDropdownvalue(applications[0])

        let theDropDown = <select className="ApplicationDropDown" onChange={(e) => handleDropDownChange(e.target.value)} defaultValue={dropdownvalue}>
            {dropDownApplicaitons.map((option, idx) => (
                <option key={idx}>{option}</option>))}
        </select>;

        // references a varible in html with "{varible}"
        return <div className='Full-Welcome'>
            <div className='Close'>
                <img
                    className='CloseImage'
                    src={CloseSVG}
                    onClick={() => handleCloseClick()}
                /></div>
            <div className='Information'>
                <img
                    src={InformationSVG}
                    onClick={() => handleInformationClick()}
                /></div>
            <h1>Welcome to FACET</h1>
            <div className='Selection-Box'>
                <table className="DirectionTable">
                    <tr><tbody>
                        <td><button className={currentTab == 0 ? 'SelectedApplicant' : 'UnselectedApplicant'} onClick={() => handleTabControl(0)}>{formatDict["dataset"].charAt(0).toUpperCase() + formatDict["dataset"].slice(1) + " Applicant"}</button></td>
                        <td><button className={currentTab == 1 ? 'SelectedApplicant' : 'UnselectedApplicant'} onClick={() => handleTabControl(1)}> Custom Application</button></td>
                    </tbody></tr></table>

                <div className="Selection-Details">

                    <div className="Application-Window" id="DivForDropDown" style={{display: currentTab == 1 ? 'none' : 'flex'}}><b>Applicants</b>
                        {theDropDown}
                        {getDetailedFeaturesOfSelected()}
                    </div>

                    <div className="Application-Window" id="DivForCustomApplicant" style={{display: currentTab == 0 ? 'none' : 'flex'}}><b>Custom Applicant</b>
                        {FeatureInputs()}

                    </div>

                    <br></br>
                    <div className="Button-Div"><button className='Confirm-Button' onClick={() => handleConfirmButton()}>Confirm</button></div>
                    <br></br>
                </div>
            </div>




        </div>;
    }
};

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

function FeatureInputTest({prettyName, name, updateValue}) {
    console.log("FeatureInput" + name)

    return (
        <div className="features-container">
            <div className='feature'>
                <div className="InlineTextFeature">{prettyName}&nbsp;</div>
                <div id={"FeatureInput" + name} className="InlineTextFeature">
                    <EditableText
                        currText={0}
                        updateValue={updateValue}
                    />
                </div>
            </div>
        </div>
    )
}

const EditableText = ({ currText, updateValue }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [text, setText] = useState(currText);

    const handleDoubleClick = () => {
        setIsEditing(true);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            setIsEditing(false);
            // Pass the updated value to the parent component
            updateValue(parseInt(text));
        }
    };

    const handleBlur = () => {
        setIsEditing(false);
        // Pass the updated value to the parent component
        updateValue(text);
    };

    const handleChange = (e) => {
        setText(e.target.value);
    };

    return (
        <div style={{ display: 'inline-block' }}>
            {isEditing ? (
                <input
                    style={{ width: Math.min(Math.max(text.length, 2), 20) + 'ch' }}
                    type="text"
                    value={text}
                    autoFocus
                    onChange={handleChange}
                    onBlur={handleBlur}
                    onKeyPress={handleKeyPress}
                />
            ) : (
                <p onClick={handleDoubleClick} style={{ cursor: 'pointer' }}>
                    {text}
                </p>
            )}
        </div>
    );
};

export default WelcomeScreen;