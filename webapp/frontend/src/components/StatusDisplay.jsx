import Feature from './Feature.jsx';
import { formatFeature, formatValue } from "../js/utilities.js";

export const StatusDisplay = ({ featureDict, formatDict, selectedInstance }) => {
    return (
        <div>
            {Object.keys(featureDict).map((key, index) => (
                <div key={index}>
                    <Feature
                        name={formatFeature(key, formatDict)}
                        value={formatValue(selectedInstance[key], key, formatDict)}
                    />
                </div>
            ))}
        </div>
    )
}

export default StatusDisplay;