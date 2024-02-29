import close from '@icons/close.svg';
import { formatFeature, formatValue } from '@src/js/utilities.js';
import { useState, useEffect } from 'react';
import './welcomescreen.css';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import InputAdornment from '@mui/material/InputAdornment';
import { StyledSwitch, StyledIconButton, StyledAvatar } from '../feature-control/StyledComponents.jsx';

const WelcomeScreen = (
    { instances,
        selectedInstance,
        setSelectedInstance,
        setIsWelcome,
        formatDict,
        featureDict,
        customApplicant,
        setCustomApplicant,
        selectCustom,
        setSelectCustom }
) => {
    const [selectedApplicant, setSelectedApplicant] = useState(selectedInstance);

    const handleConfirm = () => {
        if (!selectCustom) { // selected from drop down
            setSelectedInstance(selectedApplicant)
        } else if (selectCustom) { // custom applicant 
            setSelectedInstance(customApplicant)
        }
        setIsWelcome(false)
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

    const createDefaultApplicant = (selectedInstance) => {
        if (!selectedInstance) return null;

        const defaultApplicant = {};
        for (const key in selectedInstance) {
            defaultApplicant[key] = 0;
        }
        return defaultApplicant;
    };

    useEffect(() => {
        if (customApplicant === null) {
            // If customApplicant is null, initialize it with the default value
            const defaultApplicant = createDefaultApplicant(selectedInstance);
            setCustomApplicant(defaultApplicant);
        }
    }, [selectedInstance, customApplicant, setCustomApplicant]);
    
    // Toggle custom applicant on and off
    const handleSwitchChange = (event) => {
        setSelectCustom(event.target.checked);
    };

    // Handle changes to custom applicant fields
    const handleInputChange = (featureKey, value) => {
        setCustomApplicant({ ...customApplicant, [featureKey]: value })
        console.log("Changed Value: ", value);
    }

    // Handle Autocomplete selection change
    const handleSelectionChange = (event, newApplicant) => {
        if (newApplicant) {
            const applicantIndex = parseInt(newApplicant.split(' ')[1]);
            const selectedApplicant = instances[applicantIndex];
            setSelectedApplicant(selectedApplicant);
        }
    };


    return (
        <div className='full-welcome'>
            <div className="left-column">
                    <h1 className="welcome-title">Welcome, Applicant {selectedInstance.index}</h1>
                    <StyledSwitch
                        className='custom-switch'
                        checked={selectCustom}
                        onChange={handleSwitchChange}
                    />
                    <h2 className="custom-application-title">Custom Application</h2>
                <div className="autocomplete-container">
                    <Autocomplete 
                        className="dropdown"
                        options={instances.map((_, index) => `Applicant ${index}`)}
                        getOptionLabel={(option) => selectCustom ? 'Custom Applicant' : option}
                        onChange={handleSelectionChange}
                        renderInput={(params) => (
                            <TextField
                                {...params}
                                label={selectCustom ? 'Custom Applicant' : 'Select Applicant'}
                            />
                        )}
                        disabled={selectCustom ? true : false}
                    />
                </div>
            </div>
            <div className="right-column">
                <StyledIconButton
                    className="close-button"
                    onClick={() => setIsWelcome(false)}
                >
                    <StyledAvatar src={close} alt="close" />
                </StyledIconButton>
                <div className="feature-information">
                    <div key={selectedApplicant}>
                        {Object.keys(featureDict).map((key, index) => (
                            <div key={index} className="feature-input-container">
                                <FeatureInput
                                    featureKey={key}
                                    prettyName={formatFeature(key, formatDict)}
                                    featureValue={selectCustom ? customApplicant[key] : selectedApplicant[key]} // Check if selectedApplicant is available
                                    unit={formatDict.feature_units[formatDict.feature_names[key]]}
                                    selectCustom={selectCustom}
                                    handleInputChange={handleInputChange}
                                    selectedApplicant={selectedApplicant}
                                />
                            </div>
                        ))}
                    </div>
                </div>
                <button className='confirm-button' onClick={handleConfirm}>
                    Confirm
                </button>
            </div>
        </div>
    )
};

// displays feature input boxes
function FeatureInput({ featureKey, prettyName, featureValue, handleInputChange, selectCustom, unit, selectedApplicant }) {
    const [inputValue, setInputValue] = useState(featureValue);
    const [error, setError] = useState(false);
    const [helperText, setHelperText] = useState('');

    useEffect(() => {
        setInputValue(featureValue);
    }, [selectedApplicant, featureValue])

    const handleInputValueChange = (event) => {
        const value = event.target.value;
        setInputValue(value); // disp. input in field 

        // Validation
        if (isNaN(value) || value.startsWith('-') || value.startsWith('0')) {
            setError(true);
            setHelperText('Please enter a valid positive number.');
        } else {
            setError(false);
            setHelperText('');
        }

        // Set to custom applicant if validated 
        if (!error) {
            console.log("key: ", featureKey);
            handleInputChange(featureKey, parseInt(value));
        }
    };

    return (
        <div className='feature' style={{ marginBottom: '10px' }}>
            <TextField
                label={prettyName}
                value={inputValue}
                onChange={handleInputValueChange}
                error={error}
                helperText={helperText}
                disabled={!selectCustom}
                InputProps={{
                    endAdornment: <InputAdornment position="end" className="custom-input-adornment">{unit}</InputAdornment>,
                }}
                style={{ width: '80%' }}
            />
        </div>
    );
}
export default WelcomeScreen;