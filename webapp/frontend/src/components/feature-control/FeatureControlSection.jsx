import arrowSVG from '@icons/Arrow.svg';
import lockSVG from '@icons/Lock.svg';
import pinnedSVG from '@icons/Pinned.svg';
import unlockSVG from '@icons/UnLocked.svg';
import unPinnedSVG from '@icons/UnPinned.svg';
import { useEffect, useState } from 'react';
import { StyledAvatar, StyledIconButton, StyledSlider, StyledSwitch } from './StyledComponents.jsx';
import './feature-control.css';

const FeatureControlSection = ({ features, setFeatures, constraints, setConstraints, keepPriority, setKeepPriority, selectedScenarioIndex, setSelectedScenarioIndex }) => {
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
            if (selectedScenarioIndex !== null) {
                console.log("REACHED:");
                setSelectedScenarioIndex(null);
            }
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

            setFeatures(features);
            if (selectedScenarioIndex !== null) {
                console.log("REACHED:");
                setSelectedScenarioIndex(null);
            }

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

    const handlePriorityChange = (id, targetPriority) => {
        // id:              selected feature id 
        // target_priority: new priority value 

        // Find the feature to be updated based on the given feature ID
        const updatedFeature = features.find((feature) => feature.id === id);
        if (!updatedFeature) return;

        const currentPriority = updatedFeature.priority;

        let change = 1;
        const direction = currentPriority - targetPriority; // +++ value means current priority goes up, --- value means current priority goes down 

        const index = features.findIndex((feature) => feature.id === id);
        if (index === -1) return;

        let updatedFeatures = [...features]; // Changed from const to let
        updatedFeatures[index] = { ...updatedFeatures[index], priority: targetPriority };
        console.log(
            `Selected Feature (ID: ${id}) priority: ${currentPriority} -> ${updatedFeatures[index].priority}`
        );

        // Check Targ
        // Map over the features array to create a new array with updated priorities
        if (direction < 0) { // selected Feature moves DOWN

            updatedFeatures = updatedFeatures.map((feature) => {
                if (feature.id === id) {
                    return feature;
                }
                // check if pinned AND if in range
                else if (feature.pin_state && feature.priority > currentPriority && feature.priority <= targetPriority) {
                    //console.log(`Feature (ID: ${feature.id}) unchanged priority: ${feature.priority}`);
                    change++;
                    return feature;
                    // check if in range
                } else if (feature.priority > currentPriority && feature.priority <= targetPriority) {
                    //console.log(`Feature (ID: ${feature.id}) old priority: ${feature.priority} new priority: ${feature.priority - change}`);
                    return { ...feature, priority: feature.priority - change };
                } else {
                    return feature;
                }
            });

        } else { // selected feature moves UP

            updatedFeatures = updatedFeatures.reverse().map((feature) => {
                if (feature.id === id) {
                    return feature;
                }
                // check if pinned AND if in range
                else if (feature.pin_state && feature.priority < currentPriority && feature.priority >= targetPriority) {
                    //console.log(`Feature (ID: ${feature.id}) priority: ${feature.priority}`);
                    change++;
                    return feature;
                    // check if in range
                } else if (feature.priority < currentPriority && feature.priority >= targetPriority) {
                    //console.log(`Feature (ID: ${feature.id}}) old priority: ${feature.priority} new priority: ${feature.priority + change}`);
                    //console.log("feature priority changed ", feature.id, feature.priority + change);
                    return { ...feature, priority: feature.priority + change };
                } else {
                    return feature;
                }
            });
        }

        // Sort the updated features based on priority
        updatedFeatures.sort((a, b) => a.priority - b.priority);
        // console.log('updated features', updatedFeatures);

        // Reset the features array
        setFeatures(updatedFeatures);
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

    return (
        <div id="feature-controls-grid" className="card feature-control-tab" style={{ overflow: scroll }}>
            <div className='feature-control-header'>
                <div className="feature-control-tab-title">{feature_tab_title}</div>
                <div className='priority-toggle'>
                    <h3 className='priority-toggle'>Prioritize Features</h3>
                    <StyledSwitch
                        className='priority-toggle'
                        style={{ color: "#0090ff" }}
                        checked={keepPriority}
                        onChange={handleSwitchChange}
                    />
                </div>
            </div>
            {features.map((feature, index) => (
                <FeatureControl key={feature.id} {...feature}
                    onNumberLineChange={handleSliderConstraintChange}
                    handleSliderConstraintChange={handleSliderConstraintChange}
                    handleLockStateChange={handleLockStateChange}
                    handlePinStateChange={handlePinStateChange}
                    handlePriorityChange={handlePriorityChange}
                    keepPriority={keepPriority}
                    checkPinState={checkPinState}
                    featuresLength={features.length}
                    constraints={constraints[index]}
                />
            ))}
        </div>
    );
};

const FeatureControl = (
    { id, x, units, title, current_value, min, max, priority, lock_state, pin_state, min_range, max_range,
        handleSliderConstraintChange,
        handleLockStateChange,
        handlePinStateChange,
        handlePriorityChange,
        keepPriority,
        checkPinState,
        featuresLength,
        constraints
    }
) => {
    const [isLocked, setIsLocked] = useState(lock_state);
    const [isPinned, setIsPinned] = useState(pin_state);
    const [editedPriority, setEditedPriority] = useState(priority);
    const [range, setRange] = useState([min_range, max_range]);
    const min_distance = 1; // between min_range and max_range

    // Switch VIS state: 
    useEffect(() => {
    }, [isLocked], [isPinned]);

    useEffect(() => {
        setRange([constraints[0], constraints[1]]);
    }, [constraints]);

    useEffect(() => {
        setEditedPriority(priority);
    }, [editedPriority]);

    const handleSliderChangeCommitted = () => {
        handleSliderConstraintChange(id, range[0], range[1]);
    };

    // ARROW: Priority List Traversal 
    const handleArrowDownClick = () => {
        const target_priority = priority + 1;

        if (priority < featuresLength) {
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
        if (!isNaN(target_priority) && target_priority >= 1 && target_priority <= featuresLength && target_priority !== priority) {
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
                    disabled={isPinned || !keepPriority || priority === featuresLength}
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
                    value={range}
                    getAriaValueText={rangeText}
                    max={max}
                    min={min}
                    marks={slider_marks}
                    onChange={handleSliderChange}
                    onMouseUp={() => {
                        handleSliderChangeCommitted();
                    }}
                    valueLabelDisplay="auto"
                    disabled={isLocked}
                    disableSwap
                />
            </div>
        </div>
    );
};

export default FeatureControlSection;

