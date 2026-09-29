import { riskBand, scoreText } from '../researchDisplay'

export default function ModelInterpretation({ modelName, prediction, riskScore }) {
  const band = riskBand(riskScore)
  return <>The {modelName} model classified the transaction as{' '}
    <strong style={{ color: prediction === 'Fraudulent' ? '#C0392B' : '#1DB954' }}>{prediction}</strong>,
    {' '}with a fraud risk score of {scoreText(riskScore)}%, corresponding to a{' '}
    <strong style={{ color: band.color }}>{band.label}</strong> level.</>
}
