export default function LoadingState({ label = "Loading…" }) {
  return (
    <div className="empty-state">
      <div className="spinner" style={{ marginBottom: 12 }} />
      <div>{label}</div>
    </div>
  );
}
