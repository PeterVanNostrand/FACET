import { formatFeature, formatValue } from "../utilities.js";

export const ExplanationDisplay = ({ explanation, formatDict }) => {
    return (
        <div>
            {Object.keys(explanation).map((key, index) => (
                <div key={index}>
                    <h3>{formatFeature(key, formatDict)}</h3>
                    <p>{formatValue(explanation[key][0], key, formatDict)}, {formatValue(explanation[key][1], key, formatDict)}</p>
                </div>
            ))}
        </div>
    )
}

export default ExplanationDisplay;
