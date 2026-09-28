import "./TabBar.css"

const TabBar = ({ tabs, active, onChange }) => (
  <div className="tab-bar" role="tablist">
    {tabs.map((tab, i) => (
      <button
        key={tab.key}
        type="button"
        role="tab"
        aria-selected={active === tab.key}
        className={`tab-bar__tab${active === tab.key ? " tab-bar__tab--active" : ""}${i === 0 ? " tab-bar__tab--first" : ""}${i === tabs.length - 1 ? " tab-bar__tab--last" : ""}`}
        onClick={() => onChange(tab.key)}
      >
        {tab.label}
      </button>
    ))}
  </div>
)

export default TabBar
