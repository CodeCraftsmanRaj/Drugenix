import { Link, useLocation } from "react-router-dom";

const NAV_ITEMS = [
  { path: "/", icon: "⌂", label: "Dashboard" },
  { path: "/query", icon: "⌕", label: "New Query" },
  { path: "/history", icon: "🕘", label: "History" },
  { path: "/knowledge-graph", icon: "◈", label: "Knowledge Graph" },
  { path: "/metrics", icon: "▤", label: "Metrics" },
];

export default function Layout({ children }) {
  const location = useLocation();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">Dx</div>
          <div className="brand-text">
            <span className="brand-title">Drugenix</span>
            <span className="brand-subtitle">Agentic Drug Repurposing</span>
          </div>
        </div>

        <ul className="nav-list">
          {NAV_ITEMS.map((item) => (
            <li key={item.path}>
              <Link
                to={item.path}
                className={`nav-item ${location.pathname === item.path ? "active" : ""}`}
              >
                <span className="nav-icon">{item.icon}</span>
                {item.label}
              </Link>
            </li>
          ))}
        </ul>

        <div className="sidebar-footer">
          Drugenix v0.2 &middot; Major Project<br />
          Multi-Agent Biomedical Reasoning
        </div>
      </aside>

      <main className="main-content">{children}</main>
    </div>
  );
}
