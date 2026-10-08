import { useId } from 'react'
import './ResearchCsvLayoutDetails.css'

function ColumnList({ columns }) {
  return columns.length
    ? <ul>{columns.map(column => <li key={column}><code>{column || '(empty column name)'}</code></li>)}</ul>
    : <p className="csv-layout-none">None</p>
}

export default function ResearchCsvLayoutDetails({ issues }) {
  const id = useId()
  return <div className="csv-layout-details">
    {issues.duplicate_columns.length > 0 && <section className="csv-layout-duplicates" aria-labelledby={`${id}-duplicates`}>
      <h2 id={`${id}-duplicates`}>Duplicate column names</h2>
      <ColumnList columns={issues.duplicate_columns} />
    </section>}
    <p>Your file must match either of these formats.</p>
    <div className="csv-layout-formats">
      {[['raw', 'Raw CSV'], ['prepared', 'Preprocessed CSV']].map(([format, title]) => (
        <section className="csv-layout-format" key={format} aria-labelledby={`${id}-${format}`}>
          <h2 id={`${id}-${format}`}>{title}</h2>
          <div className="csv-layout-group">
            <h3>Missing required columns</h3>
            <ColumnList columns={issues.formats[format].missing_columns} />
          </div>
          <div className="csv-layout-group">
            <h3>Unsupported columns</h3>
            <ColumnList columns={issues.formats[format].unsupported_columns} />
          </div>
        </section>
      ))}
    </div>
  </div>
}
