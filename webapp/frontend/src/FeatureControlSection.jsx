import React, { useState, useEffect } from 'react';
import NumberLineC from './NumberLineC.jsx';
import './NumberLineC.css';
import arrowSVG from './svg/Arrow.svg';
import fillArrowSVG from './SVG/FillArrow.svg'
import lockSVG from './svg/Lock.svg';
import unlockSVG from './svg/UnLocked.svg';
import pinSVG from './svg/Pinned.svg';
import unpinSVG from './svg/UnPinned.svg';
import informationSVG from './svg/Information.svg';
const FeatureControlSection = ({ applicantInfo, fDict}) => {

    //console.log("Applicant Info", applicantInfo);
    //onsole.log("feauture Dict", fDict);

    const featureTabTitle = 'Feature Controls';
    const [feats, setFeats] = useState([]);

    useEffect(() => {
        console.log('Effect is running');
        console.log('fDict:', fDict);
        console.log('applicantInfo:', applicantInfo);

        if (fDict) {
            // populate features
            let priorityValue = 1;
    
            const newFeats = Object.entries(fDict.feature_names).map(([key, value]) => ({
                id: value,
                x: key,
                units: fDict.feature_units[value] || '',
                title: fDict.pretty_feature_names[value] || '',
                current_value: applicantInfo[key],
                min: fDict.semantic_min[value] ?? 0, // set 0 if null 
                max: fDict.semantic_max[value] ?? ((applicantInfo[key])*2), // set double current_val if null
                // Following Values not given in initial 
                priority: priorityValue++,
                lock_state: false,
                min_range: (applicantInfo[key])-((applicantInfo[key])/2),
                max_range: (applicantInfo[key])+((applicantInfo[key])/2),  // this will have bug out depending on ma; fix kater
            }));
            console.log('Feats:', newFeats); 
            setFeats(newFeats);
        }
    }, [fDict, applicantInfo]);


    // const [features, setFeatures] = useState([
    //     { id: 1, name: 'Income ($)', priority: 1, lock_state: false, numberLine: numLine1 },
    //     { id: 2, name: 'Rent ($)', priority: 2, lock_state: false, numberLine: numLine2 },
    //     { id: 3, name: 'Debt ($)', priority: 3, lock_state: false, numberLine: numLine1 },
    // ]);

    // Function to update the priority of a feature 
    const updatePriority = (featureID, priority_val, change) => {
        // featureID: ID of the Feature (that has its priority modified by user)
        // priority_val: priority value of featureID (can be changed later, not necessary to pass)
        // change: +1/-1 depending on if traversing down list (+1) or up list (-1) priority

        // Find the feature to be updated based on given feature ID
        const updatedFeature = feats.find((feature) => feature.id === featureID);

        // Check if the feature exists
        if (updatedFeature) {
            // Calculate the newPriority
            const newPriority = updatedFeature.priority + change;

            // Map over the features array to create a new array with updated priorities
            const updatedFeatures = feats.map((feature) => {
                if (feature.id === featureID) {
                    // Update the priority and lock_state of the targeted feature (remaining legacy lock_state @FIX)
                    return { ...feature, priority: newPriority, lock_state: feature.lock_state };
                } else if (feature.priority === newPriority) {
                    // If another feature has the same priority as the updated feature, adjust its priority
                    return { ...feature, priority: priority_val };
                } else {
                    // If the feature is not the one being updated or doesn't have the same priority, keep it unchanged
                    return feature;
                }
            });

            // Sort the updated features based on priority
            updatedFeatures.sort((a, b) => a.priority - b.priority);

            // Reset the features array
            setFeats(updatedFeats);
        }
    };


    const FeatureControl = ({ id, x, units, title, current_value, min, max, priority, lockState, min_range, max_range}) => {
        const [currPriority, setNewPriority] = useState(priority);
        const [isLocked, setIsLocked] = useState(lockState || false);
        const [isPinned, setIsPinned] = useState(false);

        // ARROW: Priority List Traversal 

        const handleArrowDownClick = () => {
            if (isLocked) {
                console.log(`${name}: is Locked`);
            } else {
                if (currPriority < feats.length) {
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
            //console.log('Before Lock Click:', isLocked);
            setIsLocked((prevIsLocked) => !prevIsLocked);
            //console.log('After Lock Click:', isLocked);
            console.log(name);
        };

        // Debug use: 
        useEffect(() => {
            // console.log('isLocked state changed:', isLocked);
        }, [isLocked]);

        // PIN:

        const handlePinClick = () => {
            setIsPinned((prevIsLocked) => !prevIsLocked);
            console.log('Pin Clicked!');
        };

        return (
            <div className={`feature-control-box ${isLocked ? 'locked' : ''}`}>
                <h1 className='feature-title'>{title} {units && `(${units})`}</h1>
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
                {/* Arrow SVGS commented out bc of bugs*/}
                <img className='arrow-up' onClick={handleArrowUpClick}
                    src={arrowSVG}
                    alt='arrow up'
                />
                <img className='arrow-down' onClick={handleArrowDownClick}
                    src={arrowSVG}
                    alt='arrow down'
                />
                <div className='priority-value'>{currPriority}</div>
                <div className='number-line'>
                    <NumberLineC start={min} end={max} minRange={min_range} maxRange={max_range} currentValue={current_value} />
                    {/* <NumberLineChart start={0} end={100} initialMinRange={30} initialMaxRange={50} currentValue={40} />  */}
                </div>
                <h1 className='feature-title'>{name}</h1>
            </div>
        );
    };

    return (
        <div className="feature-control-tab">
            <div className="information">
                <img src={informationSVG} />
            </div>
            <div className="feature-control-tab-title">{featureTabTitle}</div>
            {feats.map((feature) => (
                <FeatureControl key={feature.id} {...feature} />
            ))}
        </div>
    );
};

export default FeatureControlSection;
