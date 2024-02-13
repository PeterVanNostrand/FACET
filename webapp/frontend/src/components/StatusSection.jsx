import React, { useEffect, useState } from 'react';
import '../css/status-section.css'; // Import your styles.css file
import { formatFeature, formatValue } from "../js/utilities.js";


function StatusSection({ instance, status, formatDict }) {

    const [currentIndex, setCurrentIndex] = useState(1);
    const [numberOfRows, setNumberOfRows] = useState(0);

    const updateData = () => {
        const dataRow = Object.keys(instance)//instance[currentIndex];

        const rowContainer = document.getElementById('rowContainer');
        const rowContainer2 = document.getElementById('rowContainer2');

        rowContainer.innerHTML = '';
        rowContainer2.innerHTML = '';

        for (let key of dataRow) {
            if (instance.hasOwnProperty(key)) {
                const div1 = document.createElement('div');
                const div2 = document.createElement('div');

                div1.innerHTML = `<strong>${formatFeature(key, formatDict)}:</strong>`;
                div2.textContent = formatValue(instance[key], key, formatDict);;

                rowContainer.appendChild(div1);
                rowContainer.appendChild(div2);
            }
        }

        for (let key of dataRow) {
            // console.log(instance[key]); // Logging the value for debugging purposes

            if (instance.hasOwnProperty(key)) {
                // console.log("Trying to make a value appear");

                const div1 = document.createElement('div');
                const div2 = document.createElement('div');
                const dataValue = instance[key] ? instance[key] : '0';

                div1.innerHTML = `<strong>${key}:</strong>`;
                div2.textContent = dataValue;

                rowContainer2.appendChild(div1);
                rowContainer2.appendChild(div2);
            }
        }

        const approvedBox = document.querySelector('.approvedBox');
        const deniedBox = document.querySelector('.deniedBox');

        if (status === "Y") {
            approvedBox.style.display = 'block';
            deniedBox.style.display = 'none';
        } else if (status === "N") {
            approvedBox.style.display = 'none';
            deniedBox.style.display = 'block';
        }
    };

    useEffect(() => {
        console.log("Changed!")
        setNumberOfRows(Object.keys(instance).length);
        updateData();
    }, [currentIndex, instance, status]);

    return (
        <div>
            <div className="approvedBox">Your loan application has been approved</div>
            <div className="deniedBox">Unfortunately, your loan application has been denied</div>
            <div>
                <div className="Applicationbox">
                    <div className="myApplicationFlex">
                        <div className="exPopup" id="exPopup-1">
                            <div className="exOverlay"></div>
                            <div className="exContent">
                                <table style={{ width: '545px', height: '40px', borderSpacing: '10px', marginLeft: '10px' }}>
                                    <div className="row-container" id="rowContainer2"></div>
                                </table>
                            </div>
                        </div>
                    </div>
                    <span style={{ color: 'Black', marginLeft: '10px' }}>___________________________________________________________</span>
                    <div className="row-container" id="rowContainer"></div>
                </div>
            </div>
        </div>
    );
}

export default StatusSection;