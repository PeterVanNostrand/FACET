import React, { useState, useEffect } from 'react';
import NumberLineC from './NumberLineC.jsx';
import './NumberLineC.css';
import arrowSVG from './svg/Arrow.svg';
import lockSVG from './svg/Lock.svg';
import unlockSVG from './svg/UnLocked.svg';
import pinSVG from './svg/Pinned.svg';
import unpinSVG from './svg/UnPinned.svg';
import informationSVG from './svg/Information.svg';


const FeatureControlSection = ({ applicantInfo, fDict, constraints, setConstraints }) => {
    const feature_tab_title = 'Feature Controls';
    const [features, setFeatures] = useState([]);

    const handleNumberLineChange = (id, minRange, maxRange) => {
        // Find the index of the feature in constraints array
        const index = features.findIndex((feature) => feature.id === id);
        if (index !== -1) {
            // Update the constraints state
            const updatedConstraints = [...constraints];
            updatedConstraints[index] = [minRange, maxRange];
            console.log('updated', updatedConstraints)
            setConstraints(updatedConstraints);
        }
    };


    useEffect(() => {
        if (fDict) {
            // populate features
            let priorityValue = 1;

            const newFeatures = Object.entries(fDict.feature_names).map(([key, value], index) => {
                const currentValue = applicantInfo[key];
                const isZero = currentValue === 0; // checks if current feature value is zero

                const default_max = 1000;
                const default_max_range = 500;
                const default_min_range = 0;

                const lowerConstraint = constraints[index][0]
                const upperConstraint = constraints[index][1]
                console.log('lower', lowerConstraint, 'upper', upperConstraint)

                return {
                    id: value,
                    x: key,
                    units: fDict.feature_units[value] || '',
                    title: fDict.pretty_feature_names[value] || '',
                    current_value: currentValue,
                    min: fDict.semantic_min[value] ?? 0,
                    max: fDict.semantic_max[value] ?? (isZero ? default_max : currentValue * 2), // set 1 if null or double current_val if income is not 0
                    priority: priorityValue++,
                    lock_state: false,
                    pin_state: false,
                    min_range: lowerConstraint,
                    max_range: upperConstraint,
                };
            });

            console.log('Feats:', newFeatures);
            setFeatures(newFeatures);
        }
    }, [fDict, applicantInfo]);


    // Function to update the priority of a feature 
    const updatePriority = (id, change) => {
        // id: ID of the Feature (that has its priority modified by user)
        // change: +1/-1 depending on if traversing down list (+1) or up list (-1) priority

        // Find the feature to be updated based on given feature ID
        const updatedFeature = features.find((feature) => feature.id === id);
        // Check if the feature exists
        if (updatedFeature) {
            // Calculate the newPriority
            const oldPriority = updatedFeature.priority;
            const newPriority = updatedFeature.priority + change;
            // Map over the features array to create a new array with updated priorities
            const updatedFeatures = features.map((feature) => {
                if (feature.id === id) {
                    // Update the priority and lock_state of the targeted feature (remaining legacy lock_state @FIX)
                    return { ...feature, priority: newPriority, lock_state: feature.lock_state };
                } else if (feature.priority === newPriority) {
                    // If another feature has the same priority as the updated feature, adjust its priority
                    return { ...feature, priority: oldPriority };
                } else {
                    // If the feature is not the one being updated or doesn't have the same priority, keep it unchanged
                    return feature;
                }
            });

            // Sort the updated features based on priority
            updatedFeatures.sort((a, b) => a.priority - b.priority);

            // Reset the features array
            setFeatures(updatedFeatures);
        }
    };

    const changePriority = (id, target_value) => {
        // id: ID of the Feature (that has its priority modified by the user)
        // target_value: new value 

        // Find the feature to be updated based on the given feature ID
        const updatedFeature = features.find((feature) => feature.id === id);
        console.log("feature to be updated: ", updatedFeature);

        // Check if the feature exists
        if (updatedFeature) {
            // Calculate the newPriority
            const current_priority = updatedFeature.priority;
            console.log("feature's current priority: ", current_priority);
            let change = 1;
            // Calculate change
            const direction = current_priority - target_value; // +++ value means current priority goes up, --- value means current priority goes down 
            console.log("Change == ", change);
            // Map over the features array to create a new array with updated priorities
            if (direction < 0) { // Feature moves DOWN
                const updatedFeatures = features.map((feature) => {
                    if (feature.id === id) {
                        console.log("feature priority changed ", feature.id, target_value);
                        // Update the feature's priority to the target_value 
                        return { ...feature, priority: target_value };
                    }
                    else if (feature.pin_state) {
                        change++;
                    } else if (feature.priority > current_priority && feature.priority <= target_value) { // if priority is greater than current_priority and lesser than target_value, change priorty by -1 
                        // If another feature has the same priority as the updated feature, adjust its priority
                        console.log("feature priority changed ", feature.id, feature.priority - change);
                        return { ...feature, priority: feature.priority - 1 };
                    } else {
                        // Features that don't need to be changed
                        return feature;
                    }
                });

                // Sort the updated features based on priority
                updatedFeatures.sort((a, b) => a.priority - b.priority);
                console.log(updatedFeatures);
                // Reset the features array
                setFeatures(updatedFeatures);
            } else { // Feature moves UP
                const updatedFeatures = features.map((feature) => {
                    console.log("iterating over id...", feature.id);
                    console.log("Priority: ", feature.priority)
                    if (feature.id === id) {
                        // Update the priority and lock_state of the targeted feature (remaining legacy lock_state @FIX)
                        console.log("feature switching to target value: ", feature.id, target_value);
                        return { ...feature, priority: target_value };
                    } else if (feature.pin_state) {
                        change++;
                    }

                    else if (feature.priority < current_priority && feature.priority >= target_value) { // if priority is greater than current_priority and lesser than target_value, change priorty by +1 
                        // If another feature has the same priority as the updated feature, adjust its priority
                        console.log("feature priority changed ", feature.id, feature.priority + change);
                        return { ...feature, priority: feature.priority + 1 };
                    } else {
                        // Features that don't need to be changed
                        return feature;
                    }
                });

                // Sort the updated features based on priority
                updatedFeatures.sort((a, b) => a.priority - b.priority);
                console.log(updatedFeatures);
                // Reset the features array
                setFeatures(updatedFeatures);
            }
        }
    };

    const updateLockState = (id, newLockState) => {
        const updatedFeatures = features.map((feature) =>
            feature.id === id ? { ...feature, lock_state: newLockState } : feature
        );
        setFeatures(updatedFeatures);
    };

    const updatePinState = (id, newPinState) => {
        const updatedFeatures = features.map((feature) =>
            feature.id === id ? { ...feature, pin_state: newPinState } : feature
        );
        setFeatures(updatedFeatures);
    };


    const FeatureControl = ({ id, x, units, title, current_value, min, max, priority, lock_state, pin_state, min_range, max_range, onNumberLineChange }) => {
        const [currentPriority, setNewPriority] = useState(priority);
        const [isLocked, setIsLocked] = useState(lock_state);
        const [isPinned, setIsPinned] = useState(pin_state);
        const [editedPriority, setEditedPriority] = useState(priority);

        // ARROW: Priority List Traversal 

        const handleArrowDownClick = () => {
            if (isLocked) {
                console.log(`${title}: is Locked`);
            } else {
                if (currentPriority < features.length) {
                    updatePriority(id, 1);
                } else {
                    console.log('Exceeded List');
                }
            }
        };

        const handleArrowUpClick = () => {
            if (isLocked) {
                console.log(`${title}: is Locked`);
            } else {
                if (currentPriority > 1) {
                    updatePriority(id, -1);
                } else {
                    console.log('No Greater Priority');
                }
            }
        };

        //  LOCK:
        const handleLockClick = () => {
            setIsLocked((prevIsLocked) => !prevIsLocked);
            updateLockState(id, !isLocked);
            console.log("Feature: ", id, "is Locked? ", !lock_state);
        };

        // Switch VIS state: 
        useEffect(() => {
        }, [isLocked]);

        // PIN:
        const handlePinClick = () => {
            setIsPinned((prevIsPinned) => !prevIsPinned);
            updatePinState(id, !isPinned);
            console.log("Feature: ", id, "is Pinned? ", !pin_state);
        };
        // PRIORITY (inputs): 
        const handlePriorityBlur = () => {
            // Check if the edited priority is different from the current priority
            if (editedPriority !== priority) {
                // Call the function to update the priority
                changePriority(id, editedPriority);
            }
        };

        const handlePriorityChange = (event) => {
            const newValue = parseInt(event.target.value, 10);
            // Check if the new value is within valid range and different from the current priority
            if (!isNaN(newValue) && newValue >= 1 && newValue <= features.length && newValue !== priority) {
                setEditedPriority(newValue);
                // Call the function to update the priority
                changePriority(id, newValue);
            }
        };

        return (
            <div className={`feature-control-box ${isLocked ? 'locked' : ''}`}>
                <h1 className='feature-title'>{title} {units && `(${units})`}</h1>
                {/* Locks*/}
                <div className='lock'>
                    <img
                        onClick={handleLockClick}
                        className={`lock-button ${isLocked ? 'locked' : ''}`}
                        src={isLocked ? lockSVG : unlockSVG} />

                </div>
                {/* PIN functionalitiy commented out bc of bugs*/}
                <div className='pin'>
                    <img
                        src={isPinned ? pinSVG : unpinSVG}
                        alt={isPinned ? 'Pin' : 'UnPin'}
                        onClick={handlePinClick}
                        className={isPinned ? 'pinned' : ''}
                    />
                </div>
                {/* Arrows */}
                <img className='arrow-up' onClick={handleArrowUpClick}
                    src={arrowSVG}
                    alt='arrow up'
                />
                <img className='arrow-down' onClick={handleArrowDownClick}
                    src={arrowSVG}
                    alt='arrow down'
                />
                {/* Priority Value*/}
                <input className='priority-value'
                    type="number"
                    value={editedPriority}
                    onChange={handlePriorityChange}
                    onBlur={handlePriorityBlur}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                            handlePriorityBlur();
                        }
                    }}
                    style={{
                        WebkitAppearance: 'none',
                        margin: 0,
                    }}
                />
                {/* Sliders */}
                <div className='number-line'>
                    <NumberLineC id={id} start={min} end={max} minRange={min_range} maxRange={max_range} currentValue={current_value} onNumberLineChange={onNumberLineChange} />
                </div>
            </div>
        );
    };

    return (
        <div className="feature-control-tab">
            <div className="feature-control-tab-title">{feature_tab_title}</div>
            {features.map((feature) => (
                <FeatureControl key={feature.id} {...feature} onNumberLineChange={handleNumberLineChange} />
            ))}
        </div>
    );
};

export default FeatureControlSection;
