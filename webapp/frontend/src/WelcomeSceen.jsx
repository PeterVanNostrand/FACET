import React, { useEffect, useState, useRef } from 'react';
import { visualDisplay } from '../../../visualization/src/visualDisplay.js';
import * as d3 from 'd3'
import { json } from 'd3';
import './css/welcomescreen.css'
import InformationSVG from './SVG/Information.svg'

const WelcomeScreen = ({applicationList, scenarioList}) => {
    const [application, setApplication] = useState(0)
    const [currentTab, setCurrentTab] = useState(0)
    const svgRef = useRef();


    console.log("Hello World! This is running")

    const validTabNumber = (number) => {
        // TODO: Code that checks to see whether the given tab 
        //number is valid and has a corresponding tab
        return true 
    }

    // If the user clicks on a tab, 
    // - Make sure that they clicked on the tab they aren't current one
    // - Check to make sure that the number has a corresponding tab
    // - Set the Current Tab to the new tab and update
    const handleTabControl = (number) => {
        if (currentTab != number && validTabNumber(number)){
            setCurrentTab(number)
        }
    }

    const handleInformationClick = () => {

    }

    const handleConfirmButton = () => {
        // TODO: Pass either current applicant or custom applicant and scenario (from tab number)
        // and start the visualization

        //If cusstom applicant, make sure all field are filled with valid inputs before continuing 
    }

    useEffect(() => {
        const readableURL = "/visualization/data/human_readable_details.json";

        //const readableURL = "/Users/apietrick20/Documents/GitHub/FACETMQP/visualization/data/human_readable_details.json"

        // const readable = async () => await json(readableURL);

        // console.log(readable)

        fetch(readableURL)
            .then(response => response.json());
            // .then(data => console.log(data))
            // .catch(error => console.error('Error fetching JSON:', error));

        // if (explanationData && datasetDetails && readable) {
        //     const width = 800;
        //     const height = 375;

        //     const visualization = visualDisplay()
        //         .width(width)
        //         .height(height)
        //         .explanation(explanationData)
        //         .dataset_details(datasetDetails)
        //         .readable(readable)
        //         .expl_type(explType);

        //     const svg = d3.select(svgRef.current)
        //         .attr('width', width)
        //         .attr('height', height);

        //     svg.call(visualization);
        // }
    }, [
    //    explanationData, datasetDetails, readable, explType
    ]);

    return <div className='Full-Welcome'>
        <div className='Information'>
            <img
                src={InformationSVG}
                onClick={handleInformationClick}
            /></div>
        <h1><b><u>Welcome to FACET!</u></b></h1>
        <div className='Selection-Box'>
            <table className="DirectionTable">
                <tr><tbody>
                <td><button className={currentTab == 0 ? 'SelectedApplicant' : 'UnselectedApplicant'}>Applicant</button></td>
                <td><button className={currentTab == 1 ? 'SelectedApplicant' : 'UnselectedApplicant'}>Custom Application</button></td>
                </tbody></tr></table>
            Hello</div>

        <button className='Confirm-Button'>Confirrm</button>
    </div>;
};

export default WelcomeScreen;
