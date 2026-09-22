import "./FilterBar.css"
import { IoFilterOutline } from "react-icons/io5"
import { IoChevronDown } from "react-icons/io5"

const FilterBar = ({ label, value, onChange, options, disabled = false }) => {
  return (
    <div className="filterField">
      <label className={`filterBar${disabled ? " is-disabled" : ""}`}>
        <span className="visually-hidden">{label}</span>
        <IoFilterOutline className="filterIcon" aria-hidden="true" />
        <select value={value} onChange={onChange} disabled={disabled} aria-label={label}>
          {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
        <IoChevronDown className="filterChevron" aria-hidden="true" />
      </label>
    </div>
  )
}

export default FilterBar
