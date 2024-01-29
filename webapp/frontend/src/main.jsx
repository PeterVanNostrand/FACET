import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './css/index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,

)

// Status section testing
// let myInstance = {}
// myInstance["x0"] = 4895
// myInstance["x1"] = 0
// myInstance["x2"] = 10200
// myInstance["x3"] = 360
// ReactDOM.createRoot(document.getElementById('root')).render(
//   <React.StrictMode>
//     <StatusSection instance={myInstance} status="Y" />
//   </React.StrictMode>,
// )
