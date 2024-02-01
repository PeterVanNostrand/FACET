import axios, { AxiosError } from 'axios';

import React, { useEffect, useState } from 'react';
import CloseSVG from './SVG/XClose.svg';
import './css/welcomescreen.css';
import { formatFeature, formatValue } from './utilities';
import { format } from 'd3';

import webappConfig from '../../config.json';

const SERVER_URL = webappConfig.SERVER_URL
const API_PORT = webappConfig.API_PORT
const ENDPOINT = SERVER_URL + ":" + API_PORT + "/facet"

const WelcomeScreen = (
    instances,
    handleApplicationChange,
    selectedInstance,
    setSelectedInstance,
    index,
    setShowWelcomeScreen
) => {
    const [currentTab, setCurrentTab] = useState(0)
    const [dropdownvalue, setDropdownvalue] = useState(null)
    const [formatDict, setFormatDict] = useState(null)
    const [featureDict, setFeatureDict] = useState(null);

    const test = [1, 2]

    console.log("WelcomeScreen: ", selectedInstance)

    useEffect(() => {

        const pageLoad = async () => {
            let dict_data = await fetchHumanFormat()
            setFormatDict(dict_data)
            setFeatureDict(dict_data["feature_names"]);
        }

        // get the human formatting data instances
        const fetchHumanFormat = async () => {
            try {
                const response = await axios.get(ENDPOINT + "/human_format");
                return response.data
            }
            catch (error) {
                console.error(error);
                return
            }
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
        }
    }

    const handleDropDownChange = (value) => {
        setDropdownvalue(value)
        setSelectedInstance(instances.get(value))
    }

    if (formatDict == null || featureDict == null || selectedInstance == null) {
        return (
            <div className='Full-Welcome' >
                <div className='Close'>
                    <img
                        className='CloseImage'
                        src={CloseSVG}
                        onClick={() => setShowWelcomeScreen(false)}
                    />
                </div>
                <h1>Loading...</h1>
            </div>
        )
    }
    return (
        <div className='Full-Welcome' >
            <div className='Close'>
                <img
                    className='CloseImage'
                    src={CloseSVG}
                    onClick={() => setShowWelcomeScreen(false)}
                />
            </div>

            <h1>Welcome to FACET</h1>
            <div className='Selection-Box'>
                <table className="DirectionTable">
                    <tbody><tr>
                        <td>
                            <button className={currentTab == 0 ? 'SelectedApplicant' : 'UnselectedApplicant'} onClick={() => handleTabControl(0)}>
                                {formatDict["dataset"].charAt(0).toUpperCase() + formatDict["dataset"].slice(1) + " Applicant"}
                            </button>
                        </td>
                        <td>
                            <button className={currentTab == 1 ? 'SelectedApplicant' : 'UnselectedApplicant'} onClick={() => handleTabControl(1)}>
                                Custom Application
                            </button>
                        </td>
                    </tr></tbody>
                </table>

                <div className="Selection-Details">

                    <div className="Application-Window" id="DivForDropDown" style={{ display: currentTab == 1 ? 'none' : 'flex' }}>
                        <b>Applicants</b>
                        <select className="ApplicationDropDown" onChange={(e) => handleDropDownChange(e.target.value)} defaultValue={dropdownvalue}>
                            {test.map((option, idx) => (
                                <option key={idx}>{option}</option>))
                            }
                        </select>
                        <div>
                            {Object.keys(featureDict).map((key, index) => (
                                <div key={index}>
                                    <Feature name={formatFeature(key, formatDict)} value={formatValue(selectedInstance[key], key, formatDict)} />
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="Application-Window" id="DivForCustomApplicant" style={{ display: currentTab == 0 ? 'none' : 'flex' }}>
                        <b>Custom Applicant</b>
                        <div>
                            {Object.keys(featureDict).map((key, index) => (
                                <div key={index}>
                                    <FeatureInput
                                        prettyName={formatFeature(key, formatDict)}
                                        name={key}
                                        updateValue={(newValue) => (console.log(name + " has been changed to the value " + newValue))} />
                                </div>
                            ))}
                        </div>
                    </div>

                    <br></br>
                    <div className="Button-Div">
                        <button className='Confirm-Button' onClick={() => setShowWelcomeScreen(false)}>
                            Confirm
                        </button>
                    </div>
                    <br></br>
                </div>
            </div>
        </div>
    )

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

function FeatureInput({ prettyName, name, updateValue }) {
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
            {/* <input
                    //style={{ width: Math.min(Math.max(text.length, 2), 20) + 'ch' }}
                    type="text"
                    value={text}
                    autoFocus
                    onChange={handleChange}
                    onBlur={handleBlur}
                    onKeyPress={handleKeyPress}
                /> */}
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