import React, { useState, useEffect } from 'react';
import FeatureControl from './FeatureControl';
import informationSVG from './SVG/Information.svg';


const FeatureControlTab = () => {
  const featureTabTitle = 'Feature Controls';

  const [features, setFeatures] = useState([
    { id: 1, name: 'Income ($)', priority: 1, lock_state: false},
    { id: 2, name: 'Rent ($)', priority: 2, lock_state: false},
  ]);

  // const updatePriority = (featureId, oldPriority, newPriority) => {
  //   // Update the priority of the feature with the newPriority
  //   const updatedFeatures = features.map((feature) => {
  //     if (feature.priority === newPriority) {
  //       console.log(`Feature ${feature.name} updated priority to ${oldPriority}`);
  //       return { ...feature, priority: oldPriority };
  //     } else if (Math.abs(feature.priority - newPriority) === 1) {
  //       const adjustment = feature.priority > newPriority ? -1 : 1;
  //       console.log(`Feature ${feature.name} updated priority to ${feature.priority + adjustment}`);
  //       return { ...feature, priority: feature.priority + adjustment };
  //     } else {
  //       return feature;
  //     }
  //   });

  //   // Update the priority of the selected feature
  //   const updatedFeaturesFinal = updatedFeatures.map((feature) =>
  //     feature.id === featureId ? { ...feature, priority: newPriority } : feature
  //   );

  //   console.log('reach 1');

  //   // Reorder the features based on the updated priorities
  //   updatedFeaturesFinal.sort((a, b) => a.priority - b.priority);
  //   setFeatures(updatedFeaturesFinal);
  // };

//   import React from 'react';
// import FeatureControl from './FeatureControl.jsx';
// import NumberLineChart from './NumberLineChart.jsx';
// import FeatureControlTab from './FeatureControlTab.jsx';
// export function App(props) {
//   return (
//     <div className='App'>
//       {/*<NumberLine min = {0} max = {150} rangeMin= {34} rangeMax = {56} initialValue={42}/> /> */}
//       {/* <div style={{ width: '400px', height: '300px' }}>
//            <FeatureControl name='test' priority={3} />*/}
//       <FeatureControlTab />
//     </div>
//   );
// }

// Log to console
console.log('Hello console');


  const updatePriority = (featureId, oldPriority, change) => {
    // FIX: some bs happening here rn 
    // This comment suggests that there might be an issue in the code that needs fixing
    const newPriority = (oldPriority + change);
    // Update the priority of the feature with the newPriority
  
    const updatedFeatures = features.map((feature) => {
        // Locate
        if (feature.priority === newPriority) { // Locate Feature Priority that is will be updated consequently
            // Log a message indicating that the feature's priority is updated to the oldPriority
            console.log(`Feature ${feature.name} updated priority to ${oldPriority}`);
            // Return a new object with the same properties as the current feature but with updated priority
            return { ...feature, priority: oldPriority };
        } else if (Math.abs(feature.priority - newPriority) === 1) {
            // If the difference between the current priority and newPriority is 1
            const adjustment = feature.priority > newPriority ? -1 : 1;
            // Log a message indicating that the feature's priority is updated to the adjusted priority
            console.log(`Feature ${feature.name} updated priority to ${feature.priority + adjustment}`);
            // Return a new object with the same properties as the current feature but with adjusted priority
            return { ...feature, priority: feature.priority + adjustment };
        } else {
            // If the conditions are not met, return the current feature unchanged
            return feature;
        }
    });
  
    // Update the priority of the selected feature
    const updatedFeaturesFinal = updatedFeatures.map((feature) =>
        feature.id === featureId ? { ...feature, priority: newPriority } : feature
    );
    console.log('After Update - Updated State:', updatedFeaturesFinal);
    // The updatedFeaturesFinal array now contains the updated priorities, and the function is complete.
    updatedFeaturesFinal.sort((a, b) => a.priority - b.priority);
    console.log('After Sort:', updatedFeaturesFinal);
    setFeatures(updatedFeaturesFinal);
  };
  
  

// Use useEffect to log the state after the component re-renders
useEffect(() => {
  console.log('After Re-render - Current State:', features);
}, [features]);

  
  return (
    <div className="feature-control-tab">
      {/* <div className='information'> 
        <img src={informationSVG} alt="Information" />
      </div>
      <div className="feature-control-tab-title" style={{ textAlign: 'center', color: '#e28743', marginLeft: '10px' }}>
        {featureTabTitle}
      </div> */}
      {features.map((feature) => (
    <FeatureControl key={feature.id} {...feature} onUpdatePriority={updatePriority}/>
))}
    </div>
  );
};


export default FeatureControlTab;
