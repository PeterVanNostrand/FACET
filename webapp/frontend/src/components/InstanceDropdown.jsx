import React, { useState } from 'react';


export const InstanceDropdown = ({ instances, setSelectedInstance }) => {

    const [dropdownvalue, setDropdownvalue] = useState(null)

    let dropDownInstances = [];
    let applications = new Map();

    for (let i = 0; i < instances.length; i++) {
        applications.set("Application " + (i + 1), instances[i]);
        dropDownInstances.push("Application " + (i + 1));
    }

    const handleDropDownChange = (value) => {
        setDropdownvalue(value)
        setSelectedInstance(applications.get(value))
    }

    return (
        <div>
            <select className="ApplicationDropDown" onChange={(e) => handleDropDownChange(e.target.value)} defaultValue={dropdownvalue}>
                {dropDownInstances.map((option, idx) => (
                    <option key={idx}>{option}</option>))}
            </select>
        </div>
    )
}