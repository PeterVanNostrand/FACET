import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './css/index.css'
import WelcomeScreen from './WelcomeSceen.jsx'
import NumberLine from './NumberLine.jsx'
import FeatureControl from './FeatureControlTab.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <WelcomeScreen/>
    </React.StrictMode>,
)
