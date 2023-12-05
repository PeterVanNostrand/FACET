// FeatureControl.jsx
import React, { useState } from 'react'; 
import NumberLineChart from  './NumberLineChart.jsx';
import triangleSVG from './SVG/Triangle.svg';
import filltriangleSVG from './SVG/FillTriangle.svg'
import lockSVG from './SVG/Lock.svg';
import unlockSVG from './SVG/UnLocked.svg';
import pinSVG from './SVG/Pinned.svg';
import unpinSVG from './SVG/UnPinned.svg';


const FeatureControl = ({ id, name, priority, onUpdatePriority, lockState}) => {
  const [currPriority, setNewPriority] = useState(priority);
  const [isLocked, setIsLocked] = useState(lockState);
  const [isPinned, setIsPinned] = useState(false);

 // Handle Locking, Pinning and Arrows (priority)
 const handleTriangleUpClick = () => {
  if (!isPinned && currPriority !== 1) {
    console.log('up pressed');
    onUpdatePriority(id, currPriority,-1);
  }
};

const handleTriangleDownClick = () => {
  if (!isPinned) {
    console.log('down pressed');
    onUpdatePriority(id, currPriority, 1);
  }
};


 const handleLockClick = () => {
    // Toggle between locked and unlocked
    setIsLocked((prevIsLocked) => !prevIsLocked);
    console.log('Lock Clicked!');
    console.log(name);
  };

  
 const handlePinClick = () => {
  // Toggle between pinned and unpinned
  setIsPinned((prevIsLocked) => !prevIsLocked);
  console.log('Pin Clicked!');
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
          src={isPinned ? pinSVG : unpinSVG}
          alt={isPinned ? 'Pin' : 'UnPin'}
          onClick={handlePinClick}
          className={isPinned ? 'pinned' : ''}
        />
      </div>
      <div className='triangle-container'>    {/* Arrow Triangles*/}
        <img
          src={isPinned ? filltriangleSVG : triangleSVG}
          alt='Triangle up'
          onClick={handleTriangleUpClick}
        />
        <img
          src={isPinned ? filltriangleSVG : triangleSVG}
          alt='Triangle down'
          onClick={handleTriangleDownClick}
          style={{marginTop: 45, transform: 'rotate(180deg)'}} 
        />
      </div>
      <div className='priority-value'>{currPriority}</div>
      <div className='number-line'>
      <NumberLineChart start={0} end={100} initialMinRange={34} initialMaxRange={50} currentValue={54} isLocked/>
      </div>
    </div>
  );
};

export default FeatureControl;
