
export const Feature = ({ name, value }) => {
    return (
        <div className="feature">
            <p style={{ marginBottom: 0 }}>
                <b>{name}</b>: <span className="featureValue">{value}</span>
            </p>
        </div>
    )
}

export default Feature;
