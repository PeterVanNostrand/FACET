import React, { useState } from 'react';
import './styles.css'; // Import your styles.css file

function App() {
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
    fetch('loans_continuous.csv')
      .then(response => response.text())
      .then(data => {
        const rows = data.trim().split('\n');
        const headers = rows[0].split(',');
        const dataRow = rows[currentIndex].split(',');

        const rowContainer = document.getElementById('rowContainer');
        const rowContainer2 = document.getElementById('rowContainer2');
        const loanStatus = dataRow[headers.indexOf('Loan_Status')];

        rowContainer.innerHTML = '';
        rowContainer2.innerHTML = '';

        for (let i = 0; i < headers.length; i++) {
          const div = document.createElement('div');
          const dataValue = dataRow[i] ? dataRow[i] : '0';
          div.innerHTML = `<strong>${headers[i]}:</strong> ${dataValue}`;
          rowContainer.appendChild(div);
        }

        for (let i = 0; i < headers.length; i++) {
          const div = document.createElement('div');
          const dataValue = dataRow[i] ? dataRow[i] : '0';
          div.innerHTML = `<strong>${headers[i]}:</strong> ${dataValue}`;
          rowContainer2.appendChild(div);
        }

        const approvedBox = document.querySelector('.approvedBox');
        const deniedBox = document.querySelector('.deniedBox');
        const statusApp = document.querySelector('.statusApp');
        const statusDenied = document.querySelector('.statusDenied');

        const appStatus = dataRow[4];
        console.log(appStatus);

        if (String(appStatus).trim() === "Y") {
          approvedBox.style.display = 'block';
          deniedBox.style.display = 'none';
          statusApp.style.display = 'block';
          statusDenied.style.display = 'none';
        } else if (String(appStatus).trim() === "N") {
          approvedBox.style.display = 'none';
          deniedBox.style.display = 'block';
          statusApp.style.display = 'none';
          statusDenied.style.display = 'block';
        }
      })
      .catch(error => console.error('Error fetching CSV file:', error));
  };

  const movePrev = () => {
    setCurrentIndex(Math.max(1, currentIndex - 1));
    updateData();
  };

  const moveNext = () => {
    setCurrentIndex(Math.min(currentIndex + 1, numberOfRows - 1));
    updateData();
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
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', () => {
        setIsResizing(false);
        document.removeEventListener('mousemove', handleMouseMove);
      });
    }
  };

  return (
    <div>
      <div className="approvedBox">Your loan application has been approved</div>
      <div className="deniedBox">Unfortunately, your loan application has been denied</div>
      <div>
        <div className="Applicationbox">
          <div className="myApplicationFlex">
            <button className="info" id="button" onClick={togglePopup}>&#9432</button>
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

            <span style={{ color: 'Orange', marginLeft: '-23px', whiteSpace: 'nowrap' }}>My Application</span>
            <span className="statusApp">Status: Approved</span>
            <span className="statusDenied">Status: Denied</span>

            <button className="expand" id="button" onClick={exTogglePopup}>&#10530</button>
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
          <span style={{ color: 'Black', marginLeft: '10px' }}>____________________________________________________________________</span>
          <div className="row-container" id="rowContainer"></div>
          <button id="prevBtn" onClick={movePrev}>Previous</button>
          <button id="nextBtn" onClick={moveNext}>Next</button>
        </div>
      </div>
    </div>
  );
}

export default App;