export const Feature = ({ name, value }) => {
    return (
        <div className="feature">
            <p style={{marginBottom: 0}}>{name}: <span className="featureValue">{value}</span></p>
        </div>
    )
}

export default Feature;
