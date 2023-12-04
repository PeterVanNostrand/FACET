// FeatureControl.jsx
import React, { useState } from 'react'; // Don't forget to import useState
import NumberLine1 from './NumberLine1.jsx';
import triangleSVG from './SVG/Triangle.svg';
import lockSVG from './SVG/Lock.svg';
import unlockSVG from './SVG/UnLocked.svg';

const FeatureControl = ({ name, priority}) => {
  const [newPriority, setNewPriority] = useState(priority);
  const [isLocked, setIsLocked] = useState(false);
  const handleTriangleUpClick = () => {
    // Logic for on click
   //console.log('Triangle Up Clicked!');
  };

  const handleTriangleDownClick = () => {
    // logic for on click 
    //console.log('Triangle Down Clicked!');
  };

 const handleLockClick = () => {
    // Toggle between locked and unlocked
    setIsLocked((prevIsLocked) => !prevIsLocked);
    //console.log('Lock Clicked!');
  };

  return (
    <div className='feature-control-box'>
      <h1 className='feature-title'>{name}</h1>
      {/* Arrow Triangles*/}
      <div className='lock'>
        <img
        src={isLocked ? lockSVG : unlockSVG}
        alt={isLocked ? 'Lock' : 'Unlock'}
        onClick={handleLockClick}
        style={{ filter: isLocked ? 'none' : 'grayscale(100%)', color: isLocked ? '' : '#636363'}}
        />
      </div>
      <div className='triangle-container' style={{width: '25px', height: '25px'}}>
      <img src={triangleSVG} alt='Triangle up'
       style={{marginLeft: '23px', marginTop: '-5px', color: isLocked? 'none': 'black'}} onClick={handleTriangleUpClick} />
      <img src={triangleSVG} alt='Triangle down' style={{marginLeft: '23px', marginTop: '25px', transform: 'rotate(180deg)', color: isLocked? 'none': 'black'}} onClick={handleTriangleDownClick} />
      </div>
      <div className='priority-value'>{newPriority}</div>
      <div className='number-line'>
      </div>
        <NumberLine1 initialMinValue={30} initialMaxValue={50} initialCurValue={40} minX={0} maxX={150}/>
    </div>
  );
};

export default FeatureControl;
