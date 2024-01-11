import React, { useEffect, useState, useRef } from 'react';
import { visualDisplay } from '../../../visualization/src/visualDisplay.js';
import * as d3 from 'd3'
import { json } from 'd3';
import './css/welcomescreen.css'
import InformationSVG from './SVG/Information.svg'
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

const WelcomeScreen = ({applicationList, scenarioList}) => {
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

    console.log("Hello World! This is running")

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
                setInstances(response .data);
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

        if (number == 0 || number == 1){
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
        if (currentTab != number && validTabNumber(number)){
            setCurrentTab(number)
            setSelectedInstance(instances[0]);
            setDropdownvalue(dropDownApplicaitons[0])
        }
    }

    const handleInformationClick = () => {
        console.log("Information Now!");
    }

    const handleConfirmButton = () => {
        // TODO: Pass either current applicant or custom applicant and scenario (from tab number)
        // and start the visualization

        //If cusstom applicant, make sure all field are filled with valid inputs before continuing 

        setsSelected(true);
    }

    const handleDropDownChange = (value) => {
        console.log(value)
        console.log(applications)
        console.log(applications.get(value))
        setDropdownvalue(value)
        setSelectedInstance(applications.get(value))
        console.log(dropdownvalue)
        console.log(selectedInstance)
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
                <FeatureInput name={formatFeature(key, formatDict)} />
            </div>
            ))}</div>
        }

    if(isLoading){
        // If the files are still loading
        return <div></div>
    } else if (selected){
        // If a valid application has been selected
        return getDetailedFeaturesOfSelected();

    } else {
        //If an application is still being selected

        //console.log(instances)
        console.log(formatDict)
    
        let applicantDetails = "";
            
        for (let i = 0; i < instances.length; i++) {
            applications.set("Application " + (i + 1), instances[i]);
            dropDownApplicaitons.push("Application " + (i + 1));
        }

        //Suppose to set this, however, it doesn't matter since SelectedInstances changes with DropDownValue anyways
        //setDropdownvalue(applications[0])

        let theDropDown = <select className="ApplicationDropDown" onChange={(e) => handleDropDownChange(e.target.value)} defaultValue={dropdownvalue}>
                {dropDownApplicaitons.map((option, idx) => (
                    <option key={idx}>{option}</option>))}
            </select>;

        if(currentTab == 0) {
            // Pre-loaded Applications
            applicantDetails = (
                <div className="Application-Window"><b>Please choose one of the applicants in the drop down menu to use for FACET</b>
                
                {theDropDown}

                {getDetailedFeaturesOfSelected()}
            </div>
            )
        } else {
            //Custome Application 

            applicantDetails = (
                <div className="Application-Window"><b>Kindly provide the neccessary information in the given field below and ensure that all the details are completely filled out</b>

                {FeatureInputs()}

            </div>
            )
        }

        // references a varible in html with "{varible}"
        return <div className='Full-Welcome'>
            <div className='Information'>
                <img
                    src={InformationSVG}
                    onClick={() => handleInformationClick()}
                /></div>
            <h1><b><u>Welcome to FACET!</u></b></h1>
            <div className='Selection-Box'>
                <table className="DirectionTable">
                    <tr><tbody>
                    <td><button className={currentTab == 0 ? 'SelectedApplicant' : 'UnselectedApplicant'} onClick={() => handleTabControl(0)}>{formatDict["dataset"].charAt(0).toUpperCase() + formatDict["dataset"].slice(1) + " Applicant"}</button></td>
                    <td><button className={currentTab == 1 ? 'SelectedApplicant' : 'UnselectedApplicant'} onClick={() => handleTabControl(1)}> Custom Application</button></td>
                    </tbody></tr></table>

                    <div className="Selection-Details">{applicantDetails}
                    <br></br>
                    <div className="Button-Div"><button className='Confirm-Button' onClick={() => handleConfirmButton()}>Confirm</button></div>
                    <br></br>
                    </div>
                </div>

            
        </div>;
    }
};

export default WelcomeScreen;

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

function FeatureInput({ name }) {

    return (
        <div className="features-container">
            <div className="feature">
                <p>{name}: <input onChange={() => (alert("Throwing an error for input misvalidation"))}>{}</input></p>
            </div>
            {/* Add more similar div elements for each feature */}
        </div>
    )
}

// <Dropdown options={applications} value={dropdownvalue} onChange={() => handleDropDownChange()} placeholder="Select an application" /> 