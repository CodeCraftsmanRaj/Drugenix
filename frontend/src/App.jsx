import { HashRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import QueryPage from "./pages/QueryPage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import MetricsPage from "./pages/MetricsPage.jsx";
import KnowledgeGraphPage from "./pages/KnowledgeGraphPage.jsx";

export default function App() {
  return (
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/query" element={<QueryPage />} />
          <Route path="/results/:queryId" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />
        </Routes>
      </Layout>
    </HashRouter>
  );
}
