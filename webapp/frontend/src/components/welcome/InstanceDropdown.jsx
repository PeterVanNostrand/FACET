import React, { useState } from 'react';

const InstanceDropdown = (
    { instances, setSelectedInstance, dropdownIndex, setDropdownIndex }
) => {

    const handleDropDownChange = (e) => {
        const selectedIndex = e.target.selectedIndex;
        setDropdownIndex(selectedIndex);
        setSelectedInstance(instances[selectedIndex]);
    }

    return (
        <div>
            <select className="ApplicationDropDown" onChange={(e) => handleDropDownChange(e)} defaultValue={`Application ${dropdownIndex + 1}`}>
                {instances.map((instance, index) => (
                    <option key={index} value={"Application " + (index + 1)}>Application {index + 1}</option>
                ))}
            </select>
        </div>
    )
}

export default InstanceDropdown;