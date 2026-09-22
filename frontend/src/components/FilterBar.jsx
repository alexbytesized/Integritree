import "./FilterBar.css"
import { IoFilterOutline } from "react-icons/io5"
import { IoChevronDown } from "react-icons/io5"

const FilterBar = ({ label, value, onChange, options, disabled = false, hint }) => {
  const hintId = label === "Prediction outcome" ? "prediction-outcome-hint" : undefined

  return (
    <div className="filterField">
      <label className={`filterBar${disabled ? " is-disabled" : ""}`}>
        <span className="visually-hidden">{label}</span>
        <IoFilterOutline className="filterIcon" aria-hidden="true" />
        <select value={value} onChange={onChange} disabled={disabled} aria-label={label} aria-describedby={disabled ? hintId : undefined}>
          {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
        <IoChevronDown className="filterChevron" aria-hidden="true" />
      </label>
      {disabled && <small className="filterHint" id={hintId}>{hint}</small>}
    </div>
  )
}

export default FilterBar
