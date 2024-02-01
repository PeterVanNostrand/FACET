import React, { useState, useEffect } from 'react';
import FeatureControl from './FeatureControl';
import informationSVG from '../../../svg/Information.svg';
import '../../css/feature-control-style.css';

const FeatureControlTab = () => {
    const featureTabTitle = 'Feature Controls';

    const [features, setFeatures] = useState([
        // Features
        { id: 1, name: 'Income ($)', priority: 1, lock_state: false },
        { id: 2, name: 'Rent ($)', priority: 2, lock_state: false },
    ]);



    return (
        <div className="feature-control-tab">
            {/* <div className='information'> 
        <img src={informationSVG} alt="Information" />
      </div> */}
            <div className="feature-control-tab-title" style={{ textAlign: 'center', color: '#e28743',  }}>
                {featureTabTitle}
            </div>
            {features.map((feature) => (
                <FeatureControl key={feature.id} {...feature} />
            ))}
        </div>
    );
};


export default FeatureControlTab;
