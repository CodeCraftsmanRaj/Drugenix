export default function ReasoningTrace({ steps }) {
  if (!steps || steps.length === 0) {
    return <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No reasoning trace available.</div>;
  }

  return (
    <div>
      {steps.map((step, i) => (
        <div className="trace-step" key={i}>
          <div className="trace-step-number">{step.step}</div>
          <div className="trace-step-content">
            <div className="trace-step-action">{step.action}</div>
            <div className="trace-step-detail">{step.detail}</div>
            {step.source && <div className="trace-step-source">{step.source}</div>}
          </div>
        </div>
      ))}
    </div>
  );
}
