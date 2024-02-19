import React, { useEffect } from 'react';
import '../css/status-section.css'; // Import your styles.css file
import { formatFeature, formatValue } from "../js/utilities.js";


function StatusSection({ instance, formatDict, featureDict }) {

    const updateData = () => {
        const dataRow = Object.keys(instance)//instance[currentIndex];

        const valuesContainer = document.getElementById('valuesContainer');
        const statusBannerContainer = document.getElementById('statusBannerContainer');

        valuesContainer.innerHTML = '';
        statusBannerContainer.innerHTML = '';

        for (let key of dataRow) {
            if (Object.prototype.hasOwnProperty.call(instance, key)) {
                const featureDiv = document.createElement('div');
                featureDiv.className = "feature-display"

                const div1 = document.createElement('div');
                div1.className = "feature-text";
                const div2 = document.createElement('div');
                div2.className = "feature-text";

                div1.innerHTML = `<strong>${formatFeature(key, formatDict)}:</strong>`;
                div2.textContent = formatValue(instance[key], key, formatDict);

                featureDiv.append(div1);
                featureDiv.append(div2);
                valuesContainer.appendChild(featureDiv);
            }
        }

    };

    useEffect(() => {
        updateData();
    });

    return (
        <div>
            <div className='status-title-container'>
                <h2 className='status-title'>My {formatDict["scenario_terms"]["instance_name"]} </h2>
            </div>
            <div className="deniedBox">Your {formatDict["scenario_terms"]["instance_name"].toLowerCase()} has been {formatDict["scenario_terms"]["undesired_outcome"].toLowerCase()}</div>
            <div className="Applicationbox">
                <div className="myApplicationFlex">
                    <div className="exPopup" id="exPopup-1">
                        <div className="exOverlay"></div>
                        <div className="exContent">
                            <div className="row-container" id="statusBannerContainer"></div>
                        </div>
                    </div>
                </div>
                <div className="row-container" id="valuesContainer"></div>
            </div>
        </div>
    );
}

export default StatusSection;