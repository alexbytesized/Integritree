import "./FilterBar.css"
import { IoFilterOutline } from "react-icons/io5"
import { IoChevronDown } from "react-icons/io5"

const FilterBar = ({ value, onChange }) => {
  return (
    <div className="filterBar">
      <IoFilterOutline className="filterIcon" />

      <select value={value} onChange={onChange}>
        <option value="all">All Records</option>
        <option value="legitimate">Legitimate</option>
        <option value="fraudulent">Fraudulent</option>
      </select>

      <IoChevronDown className="filterChevron" />
    </div>
  )
}

export default FilterBar
