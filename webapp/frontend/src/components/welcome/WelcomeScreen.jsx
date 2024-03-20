import close from '@icons/close.svg';
import { InputLabel } from '@mui/material';
import Autocomplete from '@mui/material/Autocomplete';
import InputAdornment from '@mui/material/InputAdornment';
import TextField from '@mui/material/TextField';
import { formatFeature } from '@src/js/utilities.js';
import { useEffect, useState } from 'react';
import { StyledAvatar, StyledIconButton, StyledInput, StyledToggleButton, StyledToggleButtonGroup } from '../feature-control/StyledComponents.jsx';
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
                    {formatDict.scenario_terms.instance_name} Selection
                </h1>
                <StyledIconButton className="close-button" onClick={() => setIsWelcome(false)} style={{ margin: "16px 16px 0 0" }}>
                    <StyledAvatar src={close} alt="close" />
                </StyledIconButton>
            </div>

            <div className="welcome-body" style={{ display: 'flex' }}>
                <div className="left-column">
                    <h4 style={{ margin: "0px 0 10px", fontWeight: 600 }}>{formatDict.scenario_terms.instance_name} Type</h4>
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
                            options={instances.map((_, index) => `${formatDict.scenario_terms.instance_name} ${index}`)}
                            getOptionLabel={(option) => selectCustom ? `Custom ${formatDict.scenario_terms.instance_name}` : option}
                            onChange={handleSelectionChange}
                            renderInput={(params) => (
                                <TextField
                                    {...params}
                                    label={selectCustom ? `Custom ${formatDict.scenario_terms.instance_name}` : (applicantIndex ? `${formatDict.scenario_terms.instance_name} ${applicantIndex}` : 'Select Applicant')}
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
    const roundedValue = z;
    const [inputValue, setInputValue] = useState(roundedValue);
    const [helperText, setHelperText] = useState('');
    const [error, setError] = useState(false);
    // Define default min and max values
    const max_value = 10000
    const default_min = min ?? 0;
    const default_max = max ?? (roundedValue ? roundedValue * 2 : max_value);

    useEffect(() => {
        setInputValue(Math.round(featureValue * 100) / 100);
    }, [selectedApplicant, featureValue])

    const handleInputValueChange = (event) => {
        const value = event.target.value;
        setInputValue(value); // disp. input in field 
        console.log("Value Parse: ", parseFloat(value));

        // Validation
        if (isNaN(value) || value.trim() === '') {
            // If input is not a number or empty
            setCustomError(true);
            setError(true);
            setHelperText('Please enter a valid input.');
        } else if (value < default_min || value > default_max) {
            // If input is out of range
            setError(true);
            setCustomError(true);
            setHelperText(`Please enter a value between ${default_min} and ${default_max}`);
        } else {
            // Valid input
            setError(false);
            setCustomError(false);
            setHelperText(null);
            handleInputChange(featureKey, parseFloat(value));
        }
    }


    return (
        <div className='feature' style={{ marginBottom: '15px', width: '90%', height: '70%', position: 'relative' }}>
            <StyledInput
                label={prettyName}
                type="number"
                value={inputValue ?? 0}
                onChange={handleInputValueChange}
                error={error}
                //helperText={helperText}
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