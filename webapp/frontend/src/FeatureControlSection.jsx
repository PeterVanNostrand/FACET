import React, { useState, useEffect } from 'react';
import NumberLineChart from './NumberLineChart.jsx';
import triangleSVG from './SVG/Triangle.svg';
import filltriangleSVG from './SVG/FillTriangle.svg'
import lockSVG from './SVG/Lock.svg';
import unlockSVG from './SVG/UnLocked.svg';
import pinSVG from './SVG/Pinned.svg';
import unpinSVG from './SVG/UnPinned.svg';
import informationSVG from './SVG/Information.svg';

const FeatureControlSection = () => {
    const featureTabTitle = 'Feature Controls';

    const [features, setFeatures] = useState([
        { id: 1, name: 'Income ($)', priority: 1, lock_state: false },
        { id: 2, name: 'Rent ($)', priority: 2, lock_state: false },
        { id: 3, name: 'Debt ($)', priority: 3, lock_state: false },
    ]);

    const updatePriority = (featureID, priority_val, change) => {
        const updatedFeature = features.find((feature) => feature.id === featureID);

        if (updatedFeature) {
            const newPriority = updatedFeature.priority + change;

            const updatedFeatures = features.map((feature) => {
                if (feature.id === featureID) {
                    return { ...feature, priority: newPriority, lock_state: feature.lock_state };
                } else if (feature.priority === newPriority) {
                    //  PREFORM IF CHECK HERE
                    // IF feature is Locked: else newPrioirty += change
                    // ELSE: the return statement priority: priority-val is ok
                    return { ...feature, priority: priority_val };
                } else {
                    return feature;
                }
            });

            updatedFeatures.sort((a, b) => a.priority - b.priority);

            setFeatures(updatedFeatures);
        }
    };

    // const updateFeatureList = (features, clickedFeatureId, direction) => {
    //     const clickedFeatureIndex = features.findIndex(feature => feature.id === clickedFeatureId);

    //     // Check if the clicked feature is not found
    //     if (clickedFeatureIndex === -1) {
    //         console.error('Clicked feature not found in the list');
    //         return null;
    //     }

    //     // Calculate the new index based on the direction
    //     let newIndex = direction === 'down' ? clickedFeatureIndex + 1 : clickedFeatureIndex - 1;

    //     // Check if the new index is within bounds
    //     if (newIndex < 0 || newIndex >= features.length) {
    //         console.warn('Index out of bounds');
    //         return null;
    //     }

    //     // Swap the feature controls
    //     let newFeature = features[newIndex];

    //     // Check if the new priority is valid (not locked)
    //     while (newFeature.lock_state) {
    //         const nextIndex = direction === 'down' ? newIndex + 1 : newIndex - 1;

    //         // Check if the next index is within bounds
    //         if (nextIndex < 0 || nextIndex >= features.length) {
    //             console.warn('Index out of bounds');
    //             return null;
    //         }

    //         newIndex = nextIndex;
    //         newFeature = features[newIndex];
    //     }

    //     // Clone the features array to avoid mutating the state directly
    //     const updatedFeatures = [...features];

    //     // Swap the feature controls
    //     [updatedFeatures[clickedFeatureIndex], updatedFeatures[newIndex]] = [newFeature, features[clickedFeatureIndex]];

    //     return updatedFeatures;
    // };


    const FeatureControl = ({ id, name, priority, lockState }) => {
        const [currPriority, setNewPriority] = useState(priority);
        const [isLocked, setIsLocked] = useState(lockState || false);
        const [isPinned, setIsPinned] = useState(false);

        // ARROW: Priority List Traversal 

        const handleArrowDownClick = () => {
            if (isLocked) {
                console.log(`${name}: is Locked`);
            } else {
                if (currPriority < features.length) {
                    updatePriority(id, currPriority, 1);
                } else {
                    console.log('Exceeded List');
                }
            }
        };

        const handleArrowUpClick = () => {
            if (isLocked) {
                console.log(`${name}: is Locked`);
            } else {
                if (currPriority > 1) {
                    updatePriority(id, currPriority, -1);
                } else {
                    console.log('No Greater Priority');
                }
            }
        };

        //  LOCK:

        const handleLockClick = () => {
            console.log('Before Lock Click:', isLocked);
            setIsLocked((prevIsLocked) => !prevIsLocked);
            console.log('After Lock Click:', isLocked);
            console.log(name);
        };

        // PIN:

        useEffect(() => {
            console.log('isLocked state changed:', isLocked);
        }, [isLocked]);

        const handlePinClick = () => {
            setIsPinned((prevIsLocked) => !prevIsLocked);
            console.log('Pin Clicked!');
        };

        return (
            <div className={`feature-control-box ${isLocked ? 'locked' : ''}`}>
                <h1 className='feature-title'>{name}</h1>
                <div className='lock'>
                <button onClick={handleLockClick} className={`lock-button ${isLocked ? 'locked' : ''}`}>
                    {isLocked ? 'Unlock' : 'Lock'}
                </button>
                </div>
                {/* <div className='pin'>
                    <img
                        src={isPinned ? pinSVG : unpinSVG}
                        alt={isPinned ? 'Pin' : 'UnPin'}
                        onClick={handlePinClick}
                        className={isPinned ? 'pinned' : ''}
                    />
                </div> */}

                <button className='arrow-up' onClick={handleArrowUpClick}>
                    <span>^</span>
                    {/* <img
                        src={triangleSVG}
                        alt='Triangle up'
                    /> */}
                </button>

                <button className='arrow-down' onClick={handleArrowDownClick}>
                    <span>v</span>
                    {/* <img
                        src={triangleSVG}
                        alt='Triangle down'
                    /> */}
                </button>

                <div className='priority-value'>{currPriority}</div>
                <div className='number-line'>
                    <NumberLineChart start={0} end={100} initialMinRange={34} initialMaxRange={50} currentValue={54} isLocked />
                </div>
                <h1 className='feature-title'>{name}</h1>
            </div>
        );
    };

    return (
        <div className="feature-control-tab">
            <div className="feature-control-tab-title">{featureTabTitle}</div>
            {features.map((feature) => (
                <FeatureControl key={feature.id} {...feature} />
            ))}
        </div>
    );
};

export default FeatureControlSection;
