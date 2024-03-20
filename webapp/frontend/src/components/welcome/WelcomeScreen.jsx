import close from '@icons/close.svg';
import { InputLabel } from '@mui/material';
import Autocomplete from '@mui/material/Autocomplete';
import InputAdornment from '@mui/material/InputAdornment';
import TextField from '@mui/material/TextField';
import { formatFeature } from '@src/js/utilities.js';
import { useEffect, useState } from 'react';
import { StyledAvatar, StyledIconButton, StyledToggleButton, StyledToggleButtonGroup } from '../feature-control/StyledComponents.jsx';
import './welcome-screen.css';

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
        setSelectCustom,
        applicantIndex,
        setApplicantIndex }
) => {
    const [selectedApplicant, setSelectedApplicant] = useState(selectedInstance);
    const [customError, setCustomError] = useState(null);

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

    // Handle changes to custom applicant fields
    const handleInputChange = (featureKey, value) => {
        setCustomApplicant({ ...customApplicant, [featureKey]: value })
        console.log("Changed Value: ", value);
    }

    // Handle Autocomplete selection change
    const handleSelectionChange = (event, newApplicant) => {
        if (newApplicant) {
            const applicantIndex = parseInt(newApplicant.split(' ')[1]);
            setApplicantIndex(newApplicant.split(' ')[1]);
            const selectedApplicant = instances[applicantIndex];
            setSelectedApplicant(selectedApplicant);
        }
    };

    const handleToggleChange = (event, newSelection) => {
        setSelectCustom(newSelection === 'custom');
        if (newSelection === 'dropdown') {
            handleSelectionChange(null, `Applicant ${applicantIndex}`)
        }
    };


    return (
        <div className='full-welcome card'>
            <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                <h1 className="welcome-title">
                    Applicant Selection
                </h1>
                <StyledIconButton className="close-button" onClick={() => setIsWelcome(false)} style={{ margin: "16px 16px 0 0" }}>
                    <StyledAvatar src={close} alt="close" />
                </StyledIconButton>
            </div>

            <div className="welcome-body" style={{ display: 'flex' }}>
                <div className="left-column">
                    <h4 style={{ margin: "0px 0 10px", fontWeight: 600 }}>Applicant Type</h4>
                    <StyledToggleButtonGroup
                        sx={{
                            display: "grid",
                            gridTemplateColumns: "auto auto",
                            gridGap: "10px",
                        }}
                        value={selectCustom ? 'custom' : 'dropdown'}
                        exclusive
                        onChange={handleToggleChange}
                        aria-label="text alignment"
                        style={{
                            marginBottom: 18,
                        }}
                    >
                        <StyledToggleButton disableRipple value="dropdown">Dropdown</StyledToggleButton>
                        <StyledToggleButton disableRipple value="custom">Custom</StyledToggleButton>
                    </StyledToggleButtonGroup>
                    <div className="autocomplete-container">
                        <Autocomplete
                            className="dropdown"
                            options={instances.map((_, index) => `Applicant ${index}`)}
                            getOptionLabel={(option) => selectCustom ? 'Custom Applicant' : option}
                            onChange={handleSelectionChange}
                            renderInput={(params) => (
                                <TextField
                                    {...params}
                                    label={selectCustom ? 'Custom Applicant' : (applicantIndex ? `Applicant ${applicantIndex}` : 'Select Applicant')}
                                />
                            )}
                            disabled={selectCustom ? true : false}
                            style={{ width: "190px" }}
                            ListboxProps={{ style: { maxHeight: 250 } }}
                        />
                    </div>
                </div>
                <div className="right-column" style={{ display: 'flex', flexDirection: 'column' }}>
                    <div className="feature-information" style={{ maxHeight: 400, overflowY: "scroll" }}>
                        {Object.keys(featureDict).map((key, index) => (
                            <div key={index} className="feature-input-container">
                                <InputLabel style={{ fontWeight: 500, fontSize: 14 }}>
                                    {formatFeature(key, formatDict)}
                                </InputLabel>
                                <FeatureInput
                                    featureKey={key}
                                    featureValue={selectCustom ? customApplicant[key] : selectedApplicant[key]}
                                    unit={formatDict.feature_units[formatDict.feature_names[key]]}
                                    selectCustom={selectCustom}
                                    handleInputChange={handleInputChange}
                                    selectedApplicant={selectedApplicant}
                                    max={formatDict.feature_units[formatDict.semantic_max[key]]}
                                    min={formatDict.feature_units[formatDict.semantic_min[key]]}
                                    currentValue={formatDict.feature_units[formatDict.semantic_min[key]]} // used to gen. max when it doesn't exist
                                    setCustomError={setCustomError}
                                    customError={customError}
                                />
                            </div>
                        ))}
                    </div>
                    <div style={{ marginLeft: 'auto', marginBottom: 10, marginRight: 10, marginTop: 30 }}>
                        <button className='confirm-button' onClick={handleConfirm} disabled={customError && selectCustom}>
                            Continue
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
};

// displays feature input boxes
function FeatureInput({ featureKey, prettyName, featureValue, handleInputChange, selectCustom, unit, selectedApplicant, max, min, currentValue, setCustomError, customError }) {
    const [inputValue, setInputValue] = useState(Math.round(featureValue).toFixed(2));
    const [helperText, setHelperText] = useState('');
    // Define default min and max values
    const default_min = min ?? 0;
    const default_max = max ?? (currentValue ? currentValue * 2 : 10000);

    useEffect(() => {
        setInputValue(featureValue);
    }, [selectedApplicant, featureValue])

    const handleInputValueChange = (event) => {
        const value = event.target.value;
        setInputValue(value); // disp. input in field 

        // Validatie
        if (!isNaN(value)) {
            setInputValue(value); // display input in field 
            // Valid 
            if (value < default_min || value > default_max) {
                setError(true);
                setHelperText(`Please enter a value between ${default_min} and ${default_max}`);
            } else {
                setError(false);
                setHelperText('');
            }

            // Set to custom applicant if validated 
            if (!error) {
                console.log("key: ", featureKey);
                handleInputChange(featureKey, parseFloat(value));
            }
        }
    };

    return (
        <div className='feature' style={{ marginBottom: '15px', width: '90%', height: '70%', position: 'relative' }}>
            <TextField
                label={prettyName}
                type="number"
                value={inputValue ?? 0}
                onChange={handleInputValueChange}
                //error={error}
                disabled={!selectCustom}
                InputProps={{
                    inputProps: { step: 'any' },
                    min: default_min,
                    max: default_max,
                    endAdornment: <InputAdornment position="end" className="custom-input-adornment">{unit}</InputAdornment>,
                }}
                style={{ width: '100%', color: 'black' }}
            />
            {helperText && selectCustom && (
                <div style={{ position: 'absolute', bottom: '-15px', left: 0, color: 'red', fontSize: '0.75rem' }}>
                    {helperText}
                </div>
            )}
        </div>
    );
}
export default WelcomeScreen;