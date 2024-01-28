export const Feature = ({ name, value }) => {
    return (
        <div className="features-container">
            <div className="feature">
                <p>{name}: <span className="featureValue">{value}</span></p>
            </div>
            {/* Add more similar div elements for each feature */}
        </div>
    )
}

export default Feature;
