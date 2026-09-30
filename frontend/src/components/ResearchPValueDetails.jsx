const numberText = value => Number.isFinite(value)
  ? value.toLocaleString('en-US', { maximumSignificantDigits: 6 }) : 'Not available'
const pValueText = value => Number.isFinite(value) ? value.toExponential(6) : 'Not available'

export default function ResearchPValueDetails({ statisticalTest: test }) {
  if (!test) return <p>Test details are not available.</p>
  const exact = test.supplementary?.method === 'exact_binomial_two_sided' ? test.supplementary : null

  return <>
    <section className="info-modal-section">
      <h3>Discordant Pairs</h3>
      <p>Transactions where one model was correct and the other was incorrect.</p>
      <ul>
        <li>Total: <strong>{numberText(test.discordant_pairs)}</strong></li>
        <li>Benchmark RF correct only (B): <strong>{numberText(test.table?.rf_only_correct)}</strong></li>
        <li>RF-SMOTE correct only (C): <strong>{numberText(test.table?.smote_only_correct)}</strong></li>
      </ul>
    </section>
    <section className="info-modal-section">
      <h3>Chi-Square Distribution</h3>
      <p>The reference distribution has 1 degree of freedom. The test statistic is compared with this distribution to obtain the primary p-value.</p>
      <ul><li>Chi-square statistic: <strong>{numberText(test.statistic)}</strong></li></ul>
    </section>
    <section className="info-modal-section">
      <h3>Primary P-Value</h3>
      <ul><li>Primary p-value: <strong>{pValueText(test.p_value)}</strong></li></ul>
    </section>
    {exact && <>
      <section className="info-modal-section">
        <h3>Exact Binomial Distribution</h3>
        <p>A supplementary exact binomial test is provided when there are 1–24 discordant pairs because the chi-square approximation may be less reliable with such small counts.</p>
        <ul>
          <li>Trials: <strong>{numberText(test.discordant_pairs)}</strong></li>
          <li>Probability: <strong>0.5</strong></li>
        </ul>
      </section>
      <section className="info-modal-section">
        <h3>Supplementary P-Value</h3>
        <ul><li>Supplementary p-value: <strong>{Number.isFinite(exact.p_value) ? exact.p_value.toPrecision(6) : 'Not available'}</strong></li></ul>
      </section>
    </>}
  </>
}
