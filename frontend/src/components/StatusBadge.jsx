export default function StatusBadge({ status }) {
  const cls =
    {
      PASS: "badge-pass",
      WARN: "badge-warn",
      FAIL: "badge-fail",
    }[status] || "badge-neutral";

  return <span className={`badge ${cls}`}>{status}</span>;
}
