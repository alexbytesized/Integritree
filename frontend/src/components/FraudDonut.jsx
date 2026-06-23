import './FraudDonut.css'

const FraudDonut = ({ value, title = "Fraud Rate Percentage" }) => {
  const degree = Math.max(value * 3.6, 3); 

  return (
    <div className="fraudDonutCard">

      <div 
        className="fraudDonut"
        style={{ "--value": `${degree}deg` }}
      >
        <span>{value.toFixed(2)}%</span>
      </div>
      <p className="fraudDonutTitle">{title}</p>
    </div>
  );
};

export default FraudDonut;