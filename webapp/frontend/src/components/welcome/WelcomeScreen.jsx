
import React, { useEffect, useState } from 'react';
import CloseSVG from '../../../icons/XClose.svg';
import '../../css/welcomescreen.css';
import { formatFeature, formatValue } from '../../js/utilities';
import { InstanceDropdown } from '../InstanceDropdown';
import EditableText from '../EditableText';

const WelcomeScreen = (
    { instances,
        selectedInstance,
        setSelectedInstance,
        setIsWelcome,
        formatDict,
        featureDict
    }
) => {
    const [currentTab, setCurrentTab] = useState(0)

    const validTabNumber = (number) => {

        if (number == 0 || number == 1) {
            return true
        } else {
            return false
        }
    }

    const handleTabControl = (number) => {
        if (currentTab != number && validTabNumber(number)) {
            setCurrentTab(number)
        }
    }

    if (formatDict == null || featureDict == null || selectedInstance == null) {
        return (
            <div className='Full-Welcome' >
                <div className='Close'>
                    <img
                        className='CloseImage'
                        src={CloseSVG}
                        onClick={() => setIsWelcome(false)}
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
                    onClick={() => setIsWelcome(false)}
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
                        <InstanceDropdown
                            instances={instances}
                            setSelectedInstance={setSelectedInstance}
                        />
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
                                        updateValue={(newValue) => (console.log(name + " has been changed to the value " + newValue))}
                                    />
                                </div>
                            ))}
                        </div>
                    </div>

                    <br></br>
                    <div className="Button-Div">
                        <button className='Confirm-Button' onClick={() => setIsWelcome(false)}>
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


export default WelcomeScreen;