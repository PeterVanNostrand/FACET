
import React, { useState, useEffect } from 'react';
import close from '../../../icons/close.svg';
import '../../css/welcomescreen.css';
import { formatFeature, formatValue } from '../../js/utilities';

import InstanceDropdown from '../InstanceDropdown';
import EditableText from '../EditableText';
import Feature from '../Feature';

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
    const [customApplicant, setCustomApplicant] = useState({ ...selectedInstance })

    const handleTabChange = (tab) => {
        setCurrentTab(tab)
    }

    const handleConfirm = () => {
        if (currentTab == 0) {
            console.log("Applicant")
        } else if (currentTab == 1) {
            console.log("Custom Application")
            setSelectedInstance(customApplicant)
            setIsWelcome(false)

        }
    }

    if (formatDict == null || featureDict == null || selectedInstance == null) {
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
                                className={currentTab == 0 ? 'SelectedApplicant' : 'UnselectedApplicant'}
                                onClick={() => handleTabChange(0)}
                            >
                                {formatDict["dataset"].charAt(0).toUpperCase() + formatDict["dataset"].slice(1) + " Applicant"}
                            </button>
                        </td>
                        <td>
                            <button
                                className={currentTab == 1 ? 'SelectedApplicant' : 'UnselectedApplicant'}
                                onClick={() => handleTabChange(1)}
                            >
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
                                <FeatureInput
                                    key={index}
                                    prettyName={formatFeature(key, formatDict)}
                                    featureValue={customApplicant[key]}
                                    updateValue={(value) => {
                                        setCustomApplicant({ ...customApplicant, [key]: value })
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