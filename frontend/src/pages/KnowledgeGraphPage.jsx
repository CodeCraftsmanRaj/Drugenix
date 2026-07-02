import { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { forceCollide } from "d3-force";
import { getKgSubgraph, getKgStats } from "../api/client.js";
import LoadingState from "../components/LoadingState.jsx";

const TYPE_COLORS = {
  drug: "#4f8cff",
  target: "#34c77b",
  disease: "#f5b84e",
};

const EXAMPLE_DISEASES = [
  "Alzheimer disease", "Parkinson disease", "Type 2 diabetes",
  "COVID-19", "Influenza", "Breast cancer", "Lung cancer",
  "Hypertension", "Asthma", "Depression", "Rheumatoid arthritis",
  "HIV infection", "Cancer",
];

export default function KnowledgeGraphPage() {
  const [disease, setDisease] = useState("Alzheimer disease");
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fgRef = useRef();

  useEffect(() => {
    getKgStats().then(setStats).catch(() => {});
  }, []);

  // Apply forces whenever graph data changes
  useEffect(() => {
    if (!fgRef.current || graphData.nodes.length === 0) return;
    const fg = fgRef.current;
    fg.d3Force("charge")?.strength(-250);
    fg.d3Force("link")?.distance(90).strength(0.4);
    // Collision force prevents nodes from physically overlapping
    fg.d3Force("collide", forceCollide((node) =>
      node.type === "disease" ? 38 : node.type === "target" ? 28 : 22
    ).strength(0.9).iterations(3));
    fg.d3ReheatSimulation();
  }, [graphData]);

  async function loadGraph(d) {
    setLoading(true);
    setError(null);
    try {
      const data = await getKgSubgraph(d, 2);
      const searchedLower = d.toLowerCase().trim();

      // Keep only the searched disease node — drop other disease nodes that bleed in via shared targets
      const filteredNodes = data.nodes.filter((n) => {
        if (n.type !== "disease") return true;
        const id = (n.id || "").toLowerCase();
        const label = (n.label || "").toLowerCase();
        return id.includes(searchedLower) || searchedLower.includes(id) ||
               label.includes(searchedLower) || searchedLower.includes(label);
      });

      // Cap drug nodes at 25 to keep graph readable
      const MAX_DRUGS = 25;
      let drugCount = 0;
      const cappedNodes = filteredNodes.filter((n) => {
        if (n.type !== "drug") return true;
        drugCount++;
        return drugCount <= MAX_DRUGS;
      });

      const nodeIds = new Set(cappedNodes.map((n) => n.id));
      const filteredLinks = data.edges.filter(
        (e) => nodeIds.has(e.source) && nodeIds.has(e.target)
      );

      setGraphData({
        nodes: cappedNodes.map((n) => ({ ...n, val: n.type === "disease" ? 14 : n.type === "target" ? 9 : 6 })),
        links: filteredLinks.map((e) => ({ source: e.source, target: e.target, relation: e.relation })),
      });
    } catch (err) {
      setError(err.message);
      setGraphData({ nodes: [], links: [] });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadGraph(disease);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleSearch(e) {
    e.preventDefault();
    loadGraph(disease);
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Knowledge Graph Explorer</h1>
          <p className="page-subtitle">
            Drug → Target → Disease relationships used by the Knowledge Graph validation layer
            (NetworkX, curated + CTD-sourced data).
          </p>
        </div>
      </div>

      {stats?.chembl_loaded && (
        <div className="banner banner-info" style={{ marginBottom: 14, fontSize: 13 }}>
          ChEMBL integrated — {stats.chembl_edges?.toLocaleString()} evidence-backed drug-disease edges
          from the European Molecular Biology Laboratory's drug database (Phase 3+ clinical evidence).
        </div>
      )}

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="stat-card">
          <div className="stat-value">{stats?.total_nodes?.toLocaleString() ?? "–"}</div>
          <div className="stat-label">Total nodes</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.drugs?.toLocaleString() ?? "–"}</div>
          <div className="stat-label">Drugs</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.targets?.toLocaleString() ?? "–"}</div>
          <div className="stat-label">Targets</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.diseases ?? "–"}</div>
          <div className="stat-label">Diseases</div>
        </div>
      </div>

      {stats?.sources && (
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 14 }}>
          Data sources: {stats.sources.join(" · ")}
        </div>
      )}

      <form onSubmit={handleSearch} className="card" style={{ marginBottom: 16, display: "flex", gap: 10 }}>
        <input
          className="input"
          value={disease}
          onChange={(e) => setDisease(e.target.value)}
          placeholder="Enter a disease name…"
        />
        <button type="submit" className="btn btn-primary">Explore</button>
      </form>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
        {EXAMPLE_DISEASES.map((d) => (
          <button
            key={d}
            type="button"
            className="btn btn-ghost"
            style={{ fontSize: 12, padding: "6px 12px" }}
            onClick={() => {
              setDisease(d);
              loadGraph(d);
            }}
          >
            {d}
          </button>
        ))}
      </div>

      {error && <div className="banner banner-error">{error}</div>}

      <div className="card" style={{ padding: 0, overflow: "hidden", height: 600 }}>
        {loading ? (
          <LoadingState label="Loading graph…" />
        ) : graphData.nodes.length === 0 ? (
          <div className="empty-state" style={{ height: "100%" }}>
            <div className="empty-state-icon">◈</div>
            <div>No graph data for this disease yet.</div>
          </div>
        ) : (
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            backgroundColor="#161d2e"
            nodeLabel={(n) => `${n.label} (${n.type})`}
            nodeColor={(n) => TYPE_COLORS[n.type] || "#9aa5bd"}
            nodeVal={(n) => n.val}
            linkColor={() => "rgba(154, 165, 189, 0.25)"}
            linkDirectionalArrowLength={3}
            linkDirectionalArrowRelPos={1}
            linkLabel={(l) => l.relation}
            nodeCanvasObjectMode={() => "after"}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const isDiseaseOrTarget = node.type === "disease" || node.type === "target";
              // Always show disease + target labels; drug labels only when zoomed in
              if (!isDiseaseOrTarget && globalScale < 2.5) return;
              const label = node.label;
              const fontSize = (isDiseaseOrTarget ? 13 : 10) / globalScale;
              const padding = 2 / globalScale;
              const textWidth = ctx.measureText(label).width;
              ctx.font = `${node.type === "disease" ? "bold " : ""}${fontSize}px Inter, sans-serif`;
              // Background pill for readability
              ctx.fillStyle = "rgba(22, 29, 46, 0.82)";
              ctx.beginPath();
              ctx.roundRect(
                node.x - textWidth / 2 - padding,
                node.y + 9 / globalScale - fontSize,
                textWidth + padding * 2,
                fontSize + padding * 2,
                2 / globalScale
              );
              ctx.fill();
              ctx.fillStyle = node.type === "disease" ? "#f5b84e" : node.type === "target" ? "#34c77b" : "#c8d0e0";
              ctx.textAlign = "center";
              ctx.fillText(label, node.x, node.y + 10 / globalScale);
            }}
            d3VelocityDecay={0.3}
            cooldownTicks={150}
            onEngineStop={() => fgRef.current?.zoomToFit(600, 60)}
          />
        )}
      </div>

      <div style={{ display: "flex", gap: 20, marginTop: 14, fontSize: 12.5, color: "var(--text-secondary)" }}>
        <Legend color={TYPE_COLORS.drug} label="Drug" />
        <Legend color={TYPE_COLORS.target} label="Target" />
        <Legend color={TYPE_COLORS.disease} label="Disease" />
      </div>
    </div>
  );
}

function Legend({ color, label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ width: 10, height: 10, borderRadius: "50%", background: color, display: "inline-block" }} />
      {label}
    </div>
  );
}
