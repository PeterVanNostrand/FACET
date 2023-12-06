// FeatureControl.jsx
import React, { useState } from 'react';
import NumberLineChart from './NumberLineChart.jsx';
import triangleSVG from './SVG/Triangle.svg';
import filltriangleSVG from './SVG/FillTriangle.svg';
import lockSVG from './SVG/Lock.svg';
import unlockSVG from './SVG/UnLocked.svg';
import pinnedSVG from './SVG/Pinned.svg';
import unpinSVG from './SVG/UnPinned.svg';


const FeatureControl = ({ id, name, priority, lockState }) => {
  const [currPriority, setNewPriority] = useState(priority);
  const [isLocked, setIsLocked] = useState(lockState);
  // Commenting out isPinned state
  const [isPinned, setIsPinned] = useState(false);

  // Handle Locking, Pinning and Arrows (priority)
  const handleTriangleUpClick = () => {
    console.log('up pressed');
    // Commenting out isPinned check
    if (!isPinned && currPriority !== 1) {
      //console.log('up pressed');
      //onUpdatePriority(id, currPriority,-1);
    }
  };

  const handleTriangleDownClick = () => {
    console.log('down pressed');

  };


  const handleLockClick = () => {
    // Toggle between locked and unlocked
    setIsLocked((prevIsLocked) => !prevIsLocked);
    //console.log('Lock Clicked!');
  };


  const handlePinClick = () => {
    // Toggle between pinned and unPinned
    setIsPinned((prevIsLocked) => !prevIsLocked);
    //console.log('Pin Clicked!');
  };

  return (
    <div className={`feature-control-box ${isLocked ? 'locked' : ''}`}>
      <h1 className='feature-title'>{name}</h1>
      <div className='lock'>
        <img
          src={isLocked ? lockSVG : unlockSVG}
          alt={isLocked ? 'Lock' : 'Unlock'}
          onClick={handleLockClick}
          className={isLocked ? 'locked' : ''}
        />
      </div>
      <div className='pin'>
        <img
          //src={pinSVG}
          src={isPinned ? pinnedSVG : unpinSVG}  // Modified to always use unpinSVG
          //alt={/*isPinned ? 'Pin' : }  // Modified alt text
          onClick={handlePinClick}
        // className={isPinned ? 'pinned' : ''}  // Commented out className
        />
      </div>
      <div className='triangle-container'>    {/* Arrow Triangles*/}
        {/* Arrow Triangles */}
        <img
          src={triangleSVG}
          alt='Triangle up'
          onClick={handleTriangleUpClick}
          style={{marginLeft: 23, marginTop: -2, height: 25}}
        />
        <img
          src={triangleSVG} 
          alt='Triangle down'
          onClick={handleTriangleDownClick} // Bugged rn where this doesn't register unless .css style isn't absolute/relative
          style={{marginLeft: 23,marginTop: 18, height: 25, transform: 'rotate(180deg)'}}
        />
      </div>
      <div className='priority-value'>{currPriority}</div>
      <div className='number-line'>
        <NumberLineChart start={0} end={100} initialMinRange={34} initialMaxRange={50} currentValue={54} isLocked />
      </div>
    </div>
  );
};

export default FeatureControl;
