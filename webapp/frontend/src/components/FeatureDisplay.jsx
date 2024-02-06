export const Feature = ({ name, value }) => {
    return (
        <div className="feature">
            <p>{name}: <span className="featureValue">{value}</span></p>
        </div>
    )
}

export default Feature;
