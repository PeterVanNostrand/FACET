import { formatFeature, formatValue } from "../utilities.js";
import Feature from './FeatureDisplay.jsx';

export const StatusDisplay = ({ featureDict, formatDict, selectedInstance }) => {
    return (
        <div>
            {
                Object.keys(featureDict).map((key, index) => (
                    <div key={index}>
                        <Feature name={formatFeature(key, formatDict)} value={formatValue(selectedInstance[key], key, formatDict)} />
                    </div>
                ))
            }
        </div>
    )
}

export default StatusDisplay;