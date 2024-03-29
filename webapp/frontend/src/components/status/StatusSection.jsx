import Feature from '@src/components/Feature.jsx';
import { formatFeature, formatValue } from '@src/js/utilities';
import './status-section.css';


export const StatusSection = ({ instance, formatDict, prediction, featureDict }) => {
    // conditionally format the text and color of the status box
    let pred_text = "";
    let pred_class = "";
    if (prediction == 0) {
        pred_text = formatDict["scenario_terms"]["undesired_outcome"].toLowerCase();
        pred_class = "status-undesired";
    }
    else if (prediction == 1) {
        pred_text = formatDict["scenario_terms"]["desired_outcome"].toLowerCase();
        pred_class = "status-desired"
    }

    return (
        <div>
            {/* Card title and Status box */}
            <div id="status-header">
                <h2>My {formatDict["scenario_terms"]["instance_name"]}</h2>
                <div id="status-box" className={pred_class}>
                    Your {formatDict["scenario_terms"]["instance_name"].toLowerCase()} has been {pred_text}
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