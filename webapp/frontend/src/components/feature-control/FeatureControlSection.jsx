import arrowSVG from '@icons/Arrow.svg';
import lockSVG from '@icons/Lock.svg';
import pinnedSVG from '@icons/Pinned.svg';
import unlockSVG from '@icons/UnLocked.svg';
import unPinnedSVG from '@icons/UnPinned.svg';
import { useEffect, useState } from 'react';
import { StyledAvatar, StyledIconButton, StyledSlider, StyledSwitch } from './StyledComponents.jsx';
import './feature-control.css';

const FeatureControlSection = ({ features, setFeatures, constraints, setConstraints, keepPriority, setKeepPriority, savedScenarios, selectedScenarioIndex, setSelectedScenarioIndex }) => {
    const feature_tab_title = 'Feature Controls';

    const handleSliderConstraintChange = (id, minRange, maxRange) => {
        // Find the index of the feature in constraints array
        const index = features.findIndex((feature) => feature.id === id);
        if (index !== -1) {
            // Update the constraints state
            if (selectedScenarioIndex !== null) {
                console.log("REACHED:");
                setSelectedScenarioIndex(null);
            }
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
            //console.log(`Selected Feature (ID: ${id}) current priority:`, current_priority);

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
                    //console.log(`Selected Feature (ID: ${id}) new priority: `, updatedFeatures[index].priority);

                    updatedFeatures = updatedFeatures.map((feature) => {
                        if (feature.id === id) {
                            return feature;
                        }
                        // check if pinned AND if in range
                        else if (feature.pin_state && feature.priority > current_priority && feature.priority <= target_priority) {
                            //console.log(`Feature (ID: ${feature.id}) unchanged priority: ${feature.priority}`);
                            change++;
                            return feature;
                            // check if in range
                        } else if (feature.priority > current_priority && feature.priority <= target_priority) {
                            //console.log(`Feature (ID: ${feature.id}) old priority: ${feature.priority} new priority: ${feature.priority - change}`);
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
                    //console.log(`Selected Feature (ID: ${id}) new priority: `, updatedFeatures[index].priority);

                    // Rest of your logic for updating priorities
                    updatedFeatures = updatedFeatures.reverse().map((feature) => {
                        if (feature.id === id) {
                            return feature;
                        }
                        // check if pinned AND if in range
                        else if (feature.pin_state && feature.priority < current_priority && feature.priority >= target_priority) {
                            //console.log(`Feature (ID: ${feature.id}) priority: ${feature.priority}`);
                            change++;
                            return feature;
                            // check if in range
                        } else if (feature.priority < current_priority && feature.priority >= target_priority) {
                            //console.log(`Feature (ID: ${feature.id}}) old priority: ${feature.priority} new priority: ${feature.priority + change}`);
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
        console.log("features: ", features);
    };

    const handleScenarioChange = () => {
        const selectedScenario = savedScenarios[selectedScenarioIndex];
        const selectedFeatures = selectedScenario.features;
        setFeatures(selectedFeatures);
    }

    useEffect(() => {
        if (selectedScenarioIndex !== null) {
            handleScenarioChange();
        }
    }, [selectedScenarioIndex]);


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


    const FeatureControl = (
        { id, x, units, title, current_value, min, max, priority, lock_state, pin_state, min_range, max_range,
        }
    ) => {
        const [isLocked, setIsLocked] = useState(lock_state);
        const [isPinned, setIsPinned] = useState(pin_state);
        const [editedPriority, setEditedPriority] = useState(priority);
        const [range, setRange] = useState([min_range, max_range]);
        const min_distance = 1; // between min_range and max_range

        // useEffect(() => {
        //     if (selectedScenarioIndex !== null) {
        //         const index = parseInt(x.slice(1));
        //         console.log(`${title} id: ${x}, index: ${index}, constraints: ${constraints[index][0]}, ${constraints[index][1]}`);
        //         setRange(constraints[index][0], constraints[index][1]);
        //         handleSliderChange(id, constraints[index][0], constraints[index][1])
        //     }
        // }, [selectedScenarioIndex]);

        const handleSliderChangeCommitted = () => {
            handleSliderConstraintChange(id, range[0], range[1]);
        };
        // ARROW: Priority List Traversal 
        const handleArrowDownClick = () => {
            const target_priority = priority + 1;

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

        };

        const handleArrowUpClick = () => {
            const target_priority = priority - 1;
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
            const target_priority = parseInt(event.target.value, 10);
            setEditedPriority(target_priority);
            // Check if the edited priority is different from the current priority
            if (target_priority !== priority) {
                // Call the function to update the priority
                handlePriorityInputChange(target_priority);
            }
            else {
                // If the edited priority is the same as the previous priority, reset the input field
                setTimeout(() => {
                    setEditedPriority(priority);
                }, 200);
            }

        };

        const handlePriorityInputChange = (event) => {
            const target_priority = parseInt(event.target.value, 10);
            setEditedPriority(target_priority);
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
                    setTimeout(() => {
                        setEditedPriority(target_priority);
                        handlePriorityChange(id, target_priority);
                    }, 500);
                }
            }
            else {
                setTimeout(() => {
                    setEditedPriority(priority);
                }, 2000);
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
            <div className={`feature-control-box ${keepPriority ? '' : 'no-priority'}`}>
                <h1 className='feature-title'>{title} {units && `(${units})`}</h1>

                {/* Lock */}
                <div className='lock'>
                    <StyledIconButton
                        onClick={handleLockClick}
                        className={`lock-button ${isLocked ? 'locked' : ''}`}
                    >
                        <StyledAvatar src={isLocked ? lockSVG : unlockSVG} alt={isLocked ? 'Unlock' : 'Lock'} />
                    </StyledIconButton>
                </div>
                {/* Pin*/}
                <div className={`pin`}>
                    <StyledIconButton
                        onClick={handlePinClick}
                        className={isPinned ? 'pinned' : ''}
                        disabled={!keepPriority}
                    >
                        <StyledAvatar src={isPinned ? pinnedSVG : unPinnedSVG} alt={isPinned ? 'Unpin' : 'Pin'} />
                    </StyledIconButton>
                </div>

                {/* Arrows */}
                <div className={`arrow-up-container`}>
                    <StyledIconButton
                        onClick={handleArrowUpClick}
                        className={`arrow-up ${keepPriority ? '' : 'no-priority'}`}
                        disabled={isPinned || !keepPriority || priority === 1}
                    >
                        <StyledAvatar src={arrowSVG} alt='arrow up' />
                    </StyledIconButton>
                </div>

                <div className={`arrow-down-container`}>
                    <StyledIconButton
                        onClick={handleArrowDownClick}
                        className={`arrow-down ${keepPriority ? '' : 'no-priority'}`}
                        disabled={isPinned || !keepPriority || priority === features.length}
                    >
                        <StyledAvatar src={arrowSVG} alt='arrow down' />
                    </StyledIconButton>
                </div>

                {/* Priority Value*/}
                <input className={`priority-value priority-value-input`}
                    type={keepPriority ? "number" : "text"}
                    value={keepPriority ? editedPriority : '—'}
                    onChange={handlePriorityInputChange}
                    onBlur={handlePriorityInputBlur}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                            handlePriorityInputBlur();
                        }
                    }}
                    pattern={keepPriority ? "[0-9]*" : ''}
                    disabled={isPinned || !keepPriority}
                />

                {/* Slider */}
                <div className={`slider ${keepPriority ? 'no-priority' : ''}`}>
                    <StyledSlider
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
                <div className='priority-toggle'>
                    <h3 className='priority-toggle'>Prioritize Features</h3>
                    <StyledSwitch
                        className='priority-toggle'
                        checked={keepPriority}
                        onChange={handleSwitchChange}
                    />
                </div>
            </div>
            {features.map((feature, index) => (
                <FeatureControl key={feature.id} {...feature}
                />
            ))}
        </div>
    );
};

export default FeatureControlSection;

