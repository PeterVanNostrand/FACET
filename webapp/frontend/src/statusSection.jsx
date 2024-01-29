import React, { useState, useEffect } from 'react';
import './styles.css'; // Import your styles.css file

function StatusSection({ instance, status, formatDict, featureDict }) {
    console.log(instance);
    console.log(status);

    const [currentIndex, setCurrentIndex] = useState(1);
    const [numberOfRows, setNumberOfRows] = useState(0);
    const [isResizing, setIsResizing] = useState(false);
    const [originalWidth, setOriginalWidth] = useState(0);
    const [originalHeight, setOriginalHeight] = useState(0);

    const togglePopup = () => {
        document.getElementById("popup-1").classList.toggle("active");
    };

    const exTogglePopup = () => {
        document.getElementById("exPopup-1").classList.toggle("active");
    };

    const updateData = () => {
        const dataRow = Object.keys(instance)//instance[currentIndex];
        
        console.log(dataRow)
        console.log(instance)
        const rowContainer = document.getElementById('rowContainer');
        const rowContainer2 = document.getElementById('rowContainer2');

        rowContainer.innerHTML = '';
        rowContainer2.innerHTML = '';

        for (let key of dataRow) {
            console.log(instance[key]); // Logging the value for debugging purposes
        
            if (instance.hasOwnProperty(key)) {
                console.log("Trying to make a value appear");
        
                const div1 = document.createElement('div');
                const div2 = document.createElement('div');
                const dataValue = instance[key] ? instance[key] : '0';
        
                div1.innerHTML = `<strong>${key}:</strong>`;
                div2.textContent = dataValue;
        
                rowContainer.appendChild(div1);
                rowContainer.appendChild(div2);
            }
        }

        for (let key of dataRow) {
            console.log(instance[key]); // Logging the value for debugging purposes
        
            if (instance.hasOwnProperty(key)) {
                console.log("Trying to make a value appear");
        
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
        const statusApp = document.querySelector('.statusApp');
        const statusDenied = document.querySelector('.statusDenied');

        if (status === "Y") {
            approvedBox.style.display = 'block';
            deniedBox.style.display = 'none';
            statusApp.style.display = 'block';
            statusDenied.style.display = 'none';
        } else if (status === "N") {
            approvedBox.style.display = 'none';
            deniedBox.style.display = 'block';
            statusApp.style.display = 'none';
            statusDenied.style.display = 'block';
        }
    };

    const movePrev = () => {
        setCurrentIndex(Math.max(1, currentIndex - 1));
    };

    const moveNext = () => {
        setCurrentIndex(Math.min(currentIndex + 1, numberOfRows - 1));
    };

    const handleMouseMove = (e) => {
        if (isResizing) {
            const newWidth = originalWidth + e.clientX - exPopup.getBoundingClientRect().right;
            const newHeight = originalHeight + e.clientY - exPopup.getBoundingClientRect().bottom;

            exPopup.style.width = newWidth > 0 ? newWidth + 'px' : '0';
            exPopup.style.height = newHeight > 0 ? newHeight + 'px' : '0';
        }
    };

    const handleMouseDown = (e) => {
        if (e.target.classList.contains('resizer')) {
            setIsResizing(true);
            setOriginalWidth(exPopup.offsetWidth);
            setOriginalHeight(exPopup.offsetHeight);
        }
    };

    useEffect(() => {
        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', () => {
                setIsResizing(false);
                document.removeEventListener('mousemove', handleMouseMove);
            });
        }
    }, [isResizing, handleMouseMove]);

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
                        <button className="info" id="button" onClick={togglePopup} style={{ width: '30px', height: '30px', fill: '#D9D9D9' }}>{'\u24D8'}</button>
                        <span className="infotooltip">info</span>

                        <div className="popup" id="popup-1">
                            <div className="overlay"></div>
                            <div className="content">
                                <div className="close-btn" onClick={togglePopup}>&times;</div>
                                <p style={{ fontWeight: 'normal', fontSize: '15px' }}>
                                    This is the status section of Facet! This application displays all features used in the application and shows the status of an application.
                                </p>
                            </div>
                        </div>

                        <span style={{ color: 'Orange', marginLeft: '10px', marginTop: '3px', whiteSpace: 'nowrap' }}>My Application</span>
                        <span className="statusApp">Status: Approved</span>
                        <span className="statusDenied">Status: Denied</span>

                        <button className="expand" id="button" onClick={exTogglePopup}>{'\u2922'}</button>
                        <span className="tooltip">Expand</span>

                        <div className="exPopup" id="exPopup-1" onMouseDown={handleMouseDown}>
                            <div className="exOverlay"></div>
                            <div className="exContent">
                                <div className="exClose-btn" onClick={() => { exTogglePopup(); resetExPopupDimensions() }}>&times;</div>
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