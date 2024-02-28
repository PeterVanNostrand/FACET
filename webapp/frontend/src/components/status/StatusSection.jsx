import Feature from '@src/components/Feature.jsx';
import { formatFeature, formatValue } from '@src/js/utilities';
import './status-section.css';


export const StatusSection = ({ instance, formatDict, featureDict }) => {
    return (
        <div>
            {/* Card title and Status box */}
            <div id="status-header">
                <h2>My {formatDict["scenario_terms"]["instance_name"]}</h2>
                <div id="status-box">
                    Your {formatDict["scenario_terms"]["instance_name"].toLowerCase()} has been {formatDict["scenario_terms"]["undesired_outcome"].toLowerCase()}
                </div>
            </div>

            {/* Display of the instance's values */}
            <div id="instance-display">
                {Object.keys(featureDict).map((key, index) => (
                    <Feature
                        name={formatFeature(key, formatDict)}
                        value={formatValue(instance[key], key, formatDict)}
                        key={index}
                    />
                ))}
            </div>
        </div >
    );
}


export default StatusSection;