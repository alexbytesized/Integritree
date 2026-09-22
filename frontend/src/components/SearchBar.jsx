import "./SearchBar.css"
import { IoSearch } from "react-icons/io5"

const SearchBar = ({ value, onChange, placeholder = "Search Transactions" }) => {
  return (
    <div className="searchBar">
      <IoSearch className="searchIcon" aria-hidden="true" />

      <input type="search" aria-label="Search Transaction ID" placeholder={placeholder} value={value} onChange={onChange} />
    </div>
  )
}

export default SearchBar
