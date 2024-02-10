import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import React, { useEffect, useState } from 'react';
import arrowSVG from '../../../icons/Arrow.svg';
import lockSVG from '../../../icons/Lock.svg';
import pinSVG from '../../../icons/Pinned.svg';
import unlockSVG from '../../../icons/UnLocked.svg';
import unpinSVG from '../../../icons/UnPinned.svg';
import '../../css/feature-control.css';

const FeatureControlSection = ({ features, setFeatures, constraints, setConstraints, keepPriority, setKeepPriority }) => {
    const feature_tab_title = 'Feature Controls';

    const handleSliderConstraintChange = (id, minRange, maxRange) => {
        // Find the index of the feature in constraints array
        const index = features.findIndex((feature) => feature.id === id);
        if (index !== -1) {
            // Update the constraints state
            const updatedConstraints = [...constraints];
            updatedConstraints[index] = [minRange, maxRange];
            setConstraints(updatedConstraints);
            setFeatures(features);

            const updatedFeatures = [...features];
            updatedFeatures[index] = {
                ...updatedFeatures[index],
                min_range: minRange,
                max_range: maxRange
            };
            setFeatures(updatedFeatures);
        }
    };

    const handleLockStateChange = (id, newLockState) => {
        const index = features.findIndex((feature) => feature.id === id);

        if (index !== -1) {
            const updatedFeatures = [...features];
            updatedFeatures[index] = { ...updatedFeatures[index], lock_state: newLockState };
            setFeatures(updatedFeatures);
        }
    };

    const handlePinStateChange = (id, newPinState) => {
        const index = features.findIndex((feature) => feature.id === id);

        if (index !== -1) {
            const updatedFeatures = [...features];
            updatedFeatures[index] = { ...updatedFeatures[index], pin_state: newPinState };
            setFeatures(updatedFeatures);
        }
    };

    const handlePriorityChange = (id, target_priority) => {
        // id:              selected feature id 
        // target_priority: new priority value 

        // Find the feature to be updated based on the given feature ID
        const updatedFeature = features.find((feature) => feature.id === id);
        console.log("Selected Feature: ", updatedFeature);

        // Else: 
        if (updatedFeature) {
            const current_priority = updatedFeature.priority;
            console.log(`Selected Feature (ID: ${id}) current priority:`, current_priority);

            let change = 1;
            // Calculate change
            const direction = current_priority - target_priority; // +++ value means current priority goes up, --- value means current priority goes down 

            // Check Targ
            // Map over the features array to create a new array with updated priorities
            if (direction < 0) { // selected Feature moves DOWN
                const index = features.findIndex((feature) => feature.id === id);
                if (index !== -1) {
                    let updatedFeatures = [...features]; // Changed from const to let
                    updatedFeatures[index] = { ...updatedFeatures[index], priority: target_priority };
                    console.log(`Selected Feature (ID: ${id}) new priority: `, updatedFeatures[index].priority);

                    updatedFeatures = updatedFeatures.map((feature) => {
                        if (feature.id === id) {
                            return feature;
                        }
                        // check if pinned AND if in range
                        else if (feature.pin_state && feature.priority > current_priority && feature.priority <= target_priority) {
                            console.log(`Feature (ID: ${feature.id}) unchanged priority: ${feature.priority}`);
                            change++;
                            return feature;
                            // check if in range
                        } else if (feature.priority > current_priority && feature.priority <= target_priority) {
                            console.log(`Feature (ID: ${feature.id}) old priority: ${feature.priority} new priority: ${feature.priority - change}`);
                            return { ...feature, priority: feature.priority - change };
                        } else {
                            return feature;
                        }
                    });

                    // Sort the updated features based on priority
                    updatedFeatures.sort((a, b) => a.priority - b.priority);
                    console.log(updatedFeatures);
                    // Reset the features array
                    setFeatures(updatedFeatures);
                }
            } else { // selected eature moves UP
                const index = features.findIndex((feature) => feature.id === id);
                if (index !== -1) {
                    let updatedFeatures = [...features]; // Changed from const to let
                    updatedFeatures[index] = { ...updatedFeatures[index], priority: target_priority };
                    console.log(`Selected Feature (ID: ${id}) new priority: `, updatedFeatures[index].priority);

                    // Rest of your logic for updating priorities
                    updatedFeatures = updatedFeatures.reverse().map((feature) => {
                        if (feature.id === id) {
                            return feature;
                        }
                        // check if pinned AND if in range
                        else if (feature.pin_state && feature.priority < current_priority && feature.priority >= target_priority) {
                            console.log(`Feature (ID: ${feature.id}) priority: ${feature.priority}`);
                            change++;
                            return feature;
                            // check if in range
                        } else if (feature.priority < current_priority && feature.priority >= target_priority) {
                            console.log(`Feature (ID: ${feature.id}}) old priority: ${feature.priority} new priority: ${feature.priority + change}`);
                            //console.log("feature priority changed ", feature.id, feature.priority + change);
                            return { ...feature, priority: feature.priority + change };
                        } else {
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
        }
    };

    const checkPinState = (target_priority) => {
        const targetFeatureIndex = features.findIndex((feature) => feature.priority === target_priority);
        const targetFeature = features[targetFeatureIndex];

        if (targetFeature && targetFeature.pin_state) {
            console.log(`Target Feature (ID: ${targetFeature.id}) is Pinned`);
            return true;
        } else {
            return false;
        }
    };

    const handleSwitchChange = (event) => {
        setKeepPriority(event.target.checked);
    };

    const FeatureControl = ({ id, x, units, title, current_value, min, max, priority, lock_state, pin_state, min_range, max_range }) => {
        const [isLocked, setIsLocked] = useState(lock_state);
        const [isPinned, setIsPinned] = useState(pin_state);
        const [editedPriority, setEditedPriority] = useState(priority);
        const [range, setRange] = useState([min_range, max_range]);
        const min_distance = 1; // between min_range and max_range

        // Switch VIS state: 
        useEffect(() => {
        }, [isLocked], [isPinned]);

        useEffect(() => {
            setEditedPriority(editedPriority);
        }, [editedPriority]);

        const handleSliderChangeCommitted = () => {
            handleSliderConstraintChange(id, range[0], range[1]);
        };

        // ARROW: Priority List Traversal 
        const handleArrowDownClick = () => {
            const target_priority = priority + 1;

            if (isPinned) {
                console.log(`${title}: is Pinnned`);
            }
            else {
                if (priority < features.length) {
                    if (checkPinState(target_priority)) {
                        return;
                    }
                    else {
                        handlePriorityChange(id, target_priority);
                    }
                } else {
                    console.log('Exceeded List: no lesser priority');
                }
            }
        };

        const handleArrowUpClick = () => {
            const target_priority = priority - 1;
            if (isPinned) {
                console.log(`${title}: is Pinned`);
            } else {
                if (priority > 1) {
                    if (checkPinState(target_priority)) {
                        return;
                    }
                    else {
                        handlePriorityChange(id, target_priority);
                    }
                } else {
                    console.log('Exceeded List: no greater priority )');
                }
            }
        };

        // LOCK:
        const handleLockClick = () => {
            setIsLocked((prevIsLocked) => !prevIsLocked);
            handleLockStateChange(id, !isLocked);
            console.log(`Feature (ID: ${id}) is Locked? ${!lock_state}}`);
        };

        // PIN:
        const handlePinClick = () => {
            setIsPinned((prevIsPinned) => !prevIsPinned);
            handlePinStateChange(id, !isPinned);
            console.log(`Feature (ID: ${id}) is Pinned? ${!pin_state}}`);
        };

        // PRIORITY (inputs): 
        const handlePriorityInputBlur = (event) => {
            setEditedPriority(event);
            // Check if the edited priority is different from the current priority
            if (editedPriority !== priority) {
                // Call the function to update the priority
                handlePriorityInputChange(editedPriority);
            }
            else {
                // If the edited priority is the same as the previous priority, reset the input field
                setTimeout(() => {
                    setEditedPriority(priority);
                }, 1500);
            }

        };

        const handlePriorityInputChange = (event) => {
            setEditedPriority(event);
            const target_priority = parseInt(event.target.value, 10);
            // Check if the new value is within valid range and different from the current priority
            if (!isNaN(target_priority) && target_priority >= 1 && target_priority <= features.length && target_priority !== priority) {
                // Check target_priority pin_state
                if (checkPinState(target_priority)) {
                    setTimeout(() => {
                        setEditedPriority(priority);
                    }, 1500);
                    return;
                }
                else {
                    setEditedPriority(target_priority);
                    setTimeout(() => {
                        handlePriorityChange(id, target_priority);
                    }, 500);
                }
            }
            else {
                setTimeout(() => {
                    setEditedPriority(priority);
                }, 1500);
                return;
            }
        };

        // SLIDER: 
        const rangeText = (range) => {
            return `$${range}`;
        }

        const slider_marks = [
            {
                value: min,
                label: min,
            },
            {
                value: current_value,
                label: current_value,
            },
            {
                value: max,
                label: max,
            },
        ];

        const handleSliderChange = (event, newRange, activeThumb) => {
            if (!Array.isArray(newRange)) {
                return;
            }
            if (newRange[1] - newRange[0] < min_distance) {
                if (activeThumb === 0) {
                    const clamped = Math.min(newRange[0], max - min_distance);
                    setRange([clamped, clamped + min_distance]);
                } else {
                    const clamped = Math.max(newRange[1], min_distance);
                    setRange([clamped - min_distance, clamped]);
                }
            } else {
                setRange(newRange);
            }
        };

        return (
            <div className={`feature-control-box ${isPinned ? 'pinned' : ''}`}>
                <h1 className='feature-title'>{title} {units && `(${units})`}</h1>
                {/* Locks*/}
                <div className='lock'>
                    <img
                        onClick={handleLockClick}
                        className={`lock-button ${isLocked ? 'locked' : ''}`}
                        src={isLocked ? lockSVG : unlockSVG} />

                </div>
                {/* PIN functionalitiy commented out bc of bugs*/}
                <div className={`pin ${keepPriority ? '' : 'hidden'}`}>
                    <img
                        src={isPinned ? pinSVG : unpinSVG}
                        alt={isPinned ? 'Pin' : 'UnPin'}
                        onClick={handlePinClick}
                        className={isPinned ? 'pinned' : ''}
                    />
                </div>
                {/* Arrows */}
                <img className={`arrow-up ${keepPriority ? '' : 'hidden'}`}
                    onClick={handleArrowUpClick}
                    src={arrowSVG}
                    alt='arrow up'
                />
                <img className={`arrow-down ${keepPriority ? '' : 'hidden'}`}
                    onClick={handleArrowDownClick}
                    src={arrowSVG}
                    alt='arrow down'
                />
                {/* Priority Value*/}
                <input className={`priority-value priority-value-input ${keepPriority ? '' : 'hidden'}`}
                    type="number"
                    value={editedPriority}
                    onChange={handlePriorityInputChange}
                    onBlur={handlePriorityInputBlur}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                            handlePriorityInputBlur();
                        }
                    }}
                    pattern="[0-9]*"
                />
                {/* Sliders */}
                <div className='slider'>
                    <Slider
                        className='constraint-slider'
                        value={range}
                        onChange={handleSliderChange}
                        onMouseUp={() => {
                            handleSliderChangeCommitted();
                        }}
                        valueLabelDisplay="auto"
                        getAriaValueText={rangeText}
                        max={max}
                        min={min}
                        marks={slider_marks}
                        disabled={isLocked}
                        disableSwap
                    />
                </div>
            </div>
        );
    };


    return (
        <div className="feature-control-tab">
            <div className='feature-control-header'>
                <div className="feature-control-tab-title">{feature_tab_title}</div>
                <Switch
                    className='priority-toggle'
                    checked={keepPriority}
                    onChange={handleSwitchChange}
                />
            </div>
            {features.map((feature) => (
                <FeatureControl key={feature.id} {...feature} onNumberLineChange={handleSliderConstraintChange} />
            ))}
        </div>
    );
};

export default FeatureControlSection;
