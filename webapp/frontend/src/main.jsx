import React from 'react'
import ReactDOM from 'react-dom/client'
import StatusSection from './StatusSection.jsx'
import './css/index.css'
import './styles.css'

let myInstance = {}

myInstance["x0"] = 4895
myInstance["x1"] = 0
myInstance["x2"] = 10200
myInstance["x3"] = 360

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <StatusSection instance={myInstance} status="Y" />
  </React.StrictMode>,
)

//{instance, status, formatDict, featureDict}
//<StatusSection instance="{x0: 4895, x1: 0, x2: 10200, x3: 360}" status="Y"/>