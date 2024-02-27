import close from '@icons/close.svg';
import Feature from '@src/components/Feature.jsx';
import { formatFeature, formatValue } from '@src/js/utilities.js';
import { useState } from 'react';
import './welcomescreen.css';

import EditableText from './EditableText';
import InstanceDropdown from './InstanceDropdown';

const WelcomeScreen = (
    { instances,
        setSelectedInstance,
        setIsWelcome,
        formatDict,
        featureDict,
        applicationType,
        setApplicationType,
        dropdownIndex,
        setDropdownIndex,
        customInstance,
        setCustomInstance,
    }
) => {

    const handleTabChange = (tab) => {
        setApplicationType(tab)
    }

    const handleConfirm = () => {
        if (applicationType == "Applicant") {
            setSelectedInstance(instances[dropdownIndex])
        } else if (applicationType == "Custom") {
            setSelectedInstance(customInstance)
        }
        setIsWelcome(false)
    }

    if (formatDict == null || featureDict == null) {
        return (
            <div className='Full-Welcome' >
                <button className="tab-close" onClick={() => setIsWelcome(false)}>
                    <img src={close} alt="close" />
                </button>
                <h1>Loading...</h1>
            </div>
        )
    }
    return (
        <div className='Full-Welcome' >
            <button className="tab-close" onClick={() => setIsWelcome(false)}>
                <img src={close} alt="close" />
            </button>

            <h1>Welcome to FACET</h1>
            <div className='Selection-Box'>
                <table className="DirectionTable">
                    <tbody><tr>
                        <td>
                            <button
                                className={applicationType == 'Applicant' ? 'SelectedApplicant' : 'UnselectedApplicant'}
                                onClick={() => {
                                    setSelectedInstance(instances[dropdownIndex])
                                    handleTabChange("Applicant")
                                }}
                            >
                                {formatDict["dataset"].charAt(0).toUpperCase() + formatDict["dataset"].slice(1) + " Applicant"}
                            </button>
                        </td>
                        <td>
                            <button
                                className={applicationType == 'Custom' ? 'SelectedApplicant' : 'UnselectedApplicant'}
                                onClick={() => handleTabChange("Custom")}
                            >
                                Custom Application
                            </button>
                        </td>
                    </tr></tbody>
                </table>

                <div className="Selection-Details">

                    <div className="Application-Window" id="DivForDropDown" style={{ display: applicationType == 'Custom' ? 'none' : 'flex' }}>
                        <b>Applicants</b>
                        <InstanceDropdown
                            instances={instances}
                            setSelectedInstance={setSelectedInstance}
                            dropdownIndex={dropdownIndex}
                            setDropdownIndex={setDropdownIndex}
                        />
                        <div>
                            {Object.keys(featureDict).map((key, index) => (
                                
                                <div key={index}>
                                    <Feature name={formatFeature(key, formatDict)} value={formatValue(instances[dropdownIndex][key], key, formatDict)} />
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="Application-Window" id="DivForCustomApplicant" style={{ display: applicationType == 'Applicant' ? 'none' : 'flex' }}>
                        <b>Custom Applicant</b>
                        <div>
                            {Object.keys(featureDict).map((key, index) => (
                                <FeatureInput
                                    key={index}
                                    prettyName={formatFeature(key, formatDict)}
                                    featureValue={customInstance[key]}
                                    updateValue={(value) => {
                                        setCustomInstance({ ...customInstance, [key]: value })
                                    }}
                                />
                            ))}
                        </div>
                    </div>

                    <br></br>
                    <div className="Button-Div">
                        <button className='Confirm-Button' onClick={handleConfirm}>
                            Confirm
                        </button>
                    </div>
                    <br></br>
                </div>
            </div>
        </div>
    )

};

function FeatureInput({ prettyName, featureValue, updateValue }) {
    return (
        <div className='feature'>
            <div className="InlineTextFeature">
                {prettyName}&nbsp;
            </div>
            <EditableText
                currText={featureValue}
                updateValue={updateValue}
            />
        </div>
    )
}


export default WelcomeScreen;