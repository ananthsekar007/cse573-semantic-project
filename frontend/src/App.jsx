import { useEffect, useMemo, useState } from "react";

const CHART_WIDTH = 1040;
const CHART_HEIGHT = 700;
const CHART_MARGIN = 64;
const THEMES = ["light", "dark"];
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const TABS = [
  { id: "clustering", label: "Clustering" },
  { id: "novelty", label: "Novelty Scoring" },
  { id: "rag", label: "RAG Chat" },
];

function formatMetric(value) {
  if (value == null) return "N/A";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(3);
  }
  return String(value);
}

function extent(values) {
  return [Math.min(...values), Math.max(...values)];
}

function scale(value, domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (d0 === d1) return (r0 + r1) / 2;
  return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

async function readJsonSafely(response) {
  const text = await response.text();
  if (!text.trim()) {
    throw new Error(`Empty response body (${response.status})`);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Server returned non-JSON response (${response.status})`);
  }
}

function ThemeToggle({ theme, onToggle }) {
  return (
    <button className="theme-toggle" onClick={onToggle} type="button" aria-label="Toggle theme">
      <span className={`theme-pill ${theme === "light" ? "is-active" : ""}`}>Light</span>
      <span className={`theme-pill ${theme === "dark" ? "is-active" : ""}`}>Dark</span>
    </button>
  );
}

function TabBar({ activeTab, onSelectTab }) {
  return (
    <div className="tab-bar" role="tablist" aria-label="Dashboard sections">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`tab-button ${activeTab === tab.id ? "tab-button-active" : ""}`}
          onClick={() => onSelectTab(tab.id)}
          role="tab"
          aria-selected={activeTab === tab.id}
          type="button"
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function MetricCard({ label, value, tone = "default" }) {
  return (
    <article className={`metric-card metric-card-${tone}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{formatMetric(value)}</strong>
    </article>
  );
}

function ClusterCard({ cluster, isActive, onSelect }) {
  return (
    <button
      className={`cluster-card ${isActive ? "cluster-card-active" : ""}`}
      type="button"
      onClick={() => onSelect(cluster.cluster_id)}
    >
      <div className="cluster-card-head">
        <span className="cluster-chip" style={{ backgroundColor: cluster.color }} />
        <div>
          <p className="cluster-id">Cluster {cluster.cluster_id}</p>
          <h3>{cluster.label}</h3>
        </div>
      </div>
      <p className="cluster-meta">
        {cluster.num_patents} patents · dominant domain: {cluster.dominant_domain}
      </p>
      <div className="tag-row">
        {cluster.keywords.slice(0, 5).map((keyword) => (
          <span className="tag" key={keyword}>
            {keyword}
          </span>
        ))}
      </div>
    </button>
  );
}

function ClusterInsight({ cluster, summary, metrics }) {
  if (!cluster) return null;

  return (
    <section className="panel insight-panel">
      <div className="panel-head">
        <div>
          <p className="panel-kicker">Cluster spotlight</p>
          <h2>{cluster.label}</h2>
        </div>
        <span className="cluster-badge" style={{ borderColor: cluster.color, color: cluster.color }}>
          Cluster {cluster.cluster_id}
        </span>
      </div>

      <div className="insight-grid">
        <div className="insight-main">
          <p className="hero-copy compact-copy">
            This cluster contains {cluster.num_patents} patents and is dominated by{" "}
            <strong>{cluster.dominant_domain}</strong>. The current run uses HDBSCAN in the original embedding
            space and a 2D UMAP projection only for visualization.
          </p>

          <div className="tag-row">
            {cluster.tfidf_terms.map((term) => (
              <span className="tag strong-tag" key={term}>
                {term}
              </span>
            ))}
          </div>

          <div className="sample-list">
            {cluster.sample_patents.map((patent) => (
              <article className="sample-card" key={patent.patent_id}>
                <p className="sample-id">{patent.patent_id}</p>
                <strong>{patent.title}</strong>
                <span>{patent.domain_true}</span>
              </article>
            ))}
          </div>
        </div>

        <div className="insight-side">
          <div className="domain-breakdown">
            {Object.entries(cluster.domain_breakdown).map(([domain, count]) => {
              const pct = Math.round((count / cluster.num_patents) * 100);
              return (
                <div className="domain-bar-block" key={domain}>
                  <div className="domain-bar-row">
                    <span>{domain}</span>
                    <strong>{count}</strong>
                  </div>
                  <div className="domain-bar-track">
                    <div className="domain-bar-fill" style={{ width: `${pct}%`, backgroundColor: cluster.color }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mini-metrics">
            <MetricCard label="Clusters" value={summary.num_clusters} />
            <MetricCard label="Raw Noise" value={metrics.raw_noise_points} />
            <MetricCard label="Reassigned" value={metrics.noise_reassigned} tone="accent" />
          </div>
        </div>
      </div>
    </section>
  );
}

function ToolPreviewCard({ title, eyebrow, description, children }) {
  return (
    <section className="panel tool-panel">
      <div className="panel-head">
        <p className="panel-kicker">{eyebrow}</p>
        <h2>{title}</h2>
        <p className="hero-copy compact-copy">{description}</p>
      </div>
      <div className="tool-panel-body">{children}</div>
    </section>
  );
}

function NoveltyTab() {
  const [title, setTitle] = useState("");
  const [abstract, setAbstract] = useState("");
  const [claimText, setClaimText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/api/novelty/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          abstract,
          claim_text: claimText,
          top_k: 5,
        }),
      });
      const payload = await readJsonSafely(response);
      if (!response.ok) {
        throw new Error(payload.detail || "Novelty scoring failed");
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "Novelty scoring failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="tool-grid">
      <ToolPreviewCard
        eyebrow="Novelty workflow"
        title="Score a candidate invention against prior art"
        description="This uses the existing FAISS document index over the SBERT patent embeddings. Submit a candidate title, abstract, and representative claim to see the nearest prior-art matches from the corpus."
      >
        <form className="form-shell" onSubmit={handleSubmit}>
          <label className="form-field">
            <span className="field-label">Patent title</span>
            <input
              type="text"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Ex: Adaptive wireless scheduling for edge AI inference"
            />
          </label>
          <label className="form-field">
            <span className="field-label">Abstract or summary</span>
            <textarea
              rows="7"
              value={abstract}
              onChange={(event) => setAbstract(event.target.value)}
              placeholder="Paste the abstract or a concise invention summary here."
            />
          </label>
          <label className="form-field">
            <span className="field-label">Claim 1 or key claim</span>
            <textarea
              rows="5"
              value={claimText}
              onChange={(event) => setClaimText(event.target.value)}
              placeholder="Paste the strongest claim language for novelty comparison."
            />
          </label>
          <div className="tool-actions">
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Scoring..." : "Score novelty"}
            </button>
            <span className="status-pill">FAISS doc index</span>
          </div>
          {error ? <p className="inline-error">{error}</p> : null}
        </form>
      </ToolPreviewCard>

      <ToolPreviewCard
        eyebrow="Scoring output"
        title="Novelty result + nearest prior art"
        description="The novelty score is computed as one minus the highest similarity found in the current patent corpus."
      >
        <div className="preview-metrics">
          <MetricCard label="Novelty score" value={result?.novelty_score ?? "Pending"} tone="accent" />
          <MetricCard label="Top similarity" value={result?.top_similarity ?? "Pending"} />
          <MetricCard label="Retrieval mode" value={result?.retrieval_mode ?? "Pending"} tone="success" />
        </div>
        {result ? (
          <div className="placeholder-stack">
            <article className="placeholder-row">
              <strong>Interpretation</strong>
              <span>{result.interpretation}</span>
            </article>
            {result.matches.map((match) => (
              <article className="placeholder-row" key={`${match.patent_id}-${match.rank}`}>
                <strong>
                  {match.rank}. {match.patent_id}
                </strong>
                <span>{match.title}</span>
                <span>
                  Domain: {match.domain} · similarity: {formatMetric(match.similarity)} · novelty:{" "}
                  {formatMetric(match.novelty_score)}
                </span>
              </article>
            ))}
          </div>
        ) : (
          <div className="placeholder-stack">
            <article className="placeholder-row">
              <strong>Nearest prior art</strong>
              <span>Submit a candidate invention to see the closest patents in the corpus.</span>
            </article>
          </div>
        )}
      </ToolPreviewCard>
    </section>
  );
}

function RagTab() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE_URL}/api/rag/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_k: 6,
        }),
      });
      const payload = await readJsonSafely(response);
      if (!response.ok) {
        throw new Error(payload.detail || "RAG request failed");
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "RAG request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="tool-grid">
      <ToolPreviewCard
        eyebrow="RAG workflow"
        title="Patent chat over chunk-level retrieval"
        description="This uses the chunk-level FAISS index for retrieval. If Grok credentials are configured, the backend will call Grok for answer synthesis; otherwise it returns a grounded retrieval-only answer."
      >
        <form className="form-shell" onSubmit={handleSubmit}>
          <label className="form-field">
            <span className="field-label">Ask a patent question</span>
            <textarea
              rows="6"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ex: Which patents in this corpus discuss operator network scheduling or wireless edge optimization?"
            />
          </label>
          <div className="tool-actions">
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Retrieving..." : "Retrieve answer"}
            </button>
            <span className="status-pill">FAISS chunk index</span>
          </div>
          {error ? <p className="inline-error">{error}</p> : null}
        </form>
      </ToolPreviewCard>

      <ToolPreviewCard
        eyebrow="Answer surface"
        title="Grounded response + citations"
        description="The answer is paired with retrieved chunks and source patents so users can inspect the supporting evidence."
      >
        {result ? (
          <>
            <div className="preview-metrics">
              <MetricCard label="Answer mode" value={result.mode} tone="accent" />
              <MetricCard label="Retrieval mode" value={result.retrieval_mode} />
              <MetricCard label="Chunks retrieved" value={result.retrieved_chunks.length} />
              <MetricCard label="Supporting patents" value={result.supporting_patents.length} tone="success" />
            </div>
            <div className="placeholder-stack">
              <article className="placeholder-row">
                <strong>Generated answer</strong>
                <span>{result.answer}</span>
              </article>
              {result.retrieved_chunks.map((chunk) => (
                <article className="placeholder-row" key={`${chunk.patent_id}-${chunk.chunk_index}`}>
                  <strong>
                    {chunk.patent_id} · chunk {chunk.chunk_index} · score {formatMetric(chunk.similarity)}
                  </strong>
                  <span>{chunk.title}</span>
                  <span>{chunk.text}</span>
                </article>
              ))}
            </div>
          </>
        ) : (
          <div className="placeholder-stack">
            <article className="placeholder-row">
              <strong>Generated answer</strong>
              <span>Ask a question to retrieve supporting chunks and generate a grounded answer.</span>
            </article>
          </div>
        )}
      </ToolPreviewCard>
    </section>
  );
}

function ClusterPlot({ points, clusters, activeClusterId, onSelectCluster }) {
  const [hovered, setHovered] = useState(null);

  const visiblePoints = useMemo(() => {
    if (activeClusterId == null) return points;
    return points;
  }, [points, activeClusterId]);

  const xValues = visiblePoints.map((point) => point.umap_x);
  const yValues = visiblePoints.map((point) => point.umap_y);
  const xDomain = extent(xValues);
  const yDomain = extent(yValues);

  return (
    <div className="plot-shell">
      <svg className="plot" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label="UMAP cluster scatter plot">
        <defs>
          <linearGradient id="plotWash" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.82)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.28)" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width={CHART_WIDTH} height={CHART_HEIGHT} rx="30" className="plot-frame" />
        <rect x="0" y="0" width={CHART_WIDTH} height={CHART_HEIGHT} rx="30" fill="url(#plotWash)" />

        {[0, 1, 2, 3, 4].map((tick) => {
          const x = CHART_MARGIN + ((CHART_WIDTH - 2 * CHART_MARGIN) / 4) * tick;
          const y = CHART_MARGIN + ((CHART_HEIGHT - 2 * CHART_MARGIN) / 4) * tick;
          return (
            <g key={tick}>
              <line x1={x} y1={CHART_MARGIN} x2={x} y2={CHART_HEIGHT - CHART_MARGIN} className="grid-line" />
              <line x1={CHART_MARGIN} y1={y} x2={CHART_WIDTH - CHART_MARGIN} y2={y} className="grid-line" />
            </g>
          );
        })}

        <line
          x1={CHART_MARGIN}
          y1={CHART_HEIGHT - CHART_MARGIN}
          x2={CHART_WIDTH - CHART_MARGIN}
          y2={CHART_HEIGHT - CHART_MARGIN}
          className="axis-line"
        />
        <line
          x1={CHART_MARGIN}
          y1={CHART_MARGIN}
          x2={CHART_MARGIN}
          y2={CHART_HEIGHT - CHART_MARGIN}
          className="axis-line"
        />

        {points.map((point) => {
          const cx = scale(point.umap_x, xDomain, [CHART_MARGIN, CHART_WIDTH - CHART_MARGIN]);
          const cy = scale(point.umap_y, yDomain, [CHART_HEIGHT - CHART_MARGIN, CHART_MARGIN]);
          const isActive = activeClusterId == null || point.cluster_pred === activeClusterId;
          const isDimmed = activeClusterId != null && point.cluster_pred !== activeClusterId;

          return (
            <circle
              key={point.patent_id}
              cx={cx}
              cy={cy}
              r={hovered?.patent_id === point.patent_id ? 9 : 7}
              fill={point.color}
              fillOpacity={isDimmed ? 0.16 : 0.88}
              stroke={isActive ? "rgba(12, 9, 7, 0.45)" : "rgba(12, 9, 7, 0.08)"}
              strokeWidth={isActive ? "1" : "0.6"}
              onMouseEnter={() => setHovered({ ...point, cx, cy })}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onSelectCluster(point.cluster_pred)}
            />
          );
        })}

        <text x={CHART_WIDTH / 2} y={CHART_HEIGHT - 18} className="axis-label">
          UMAP-1
        </text>
        <text x="18" y={CHART_HEIGHT / 2} className="axis-label axis-label-y">
          UMAP-2
        </text>

        {hovered ? (
          <g transform={`translate(${Math.min(hovered.cx + 16, CHART_WIDTH - 280)}, ${Math.max(hovered.cy - 132, 18)})`}>
            <rect width="246" height="118" rx="16" className="tooltip-box" />
            <text x="14" y="25" className="tooltip-title">
              {hovered.patent_id}
            </text>
            <text x="14" y="48" className="tooltip-copy">
              {hovered.title.slice(0, 36)}
            </text>
            <text x="14" y="69" className="tooltip-copy">
              domain: {hovered.domain_true}
            </text>
            <text x="14" y="90" className="tooltip-copy">
              cluster: {hovered.cluster_pred} {hovered.was_noise ? "(reassigned)" : ""}
            </text>
          </g>
        ) : null}
      </svg>

      <div className="legend-row">
        {clusters.map((cluster) => (
          <button
            className={`legend-pill ${activeClusterId === cluster.cluster_id ? "legend-pill-active" : ""}`}
            key={cluster.cluster_id}
            onClick={() => onSelectCluster(activeClusterId === cluster.cluster_id ? null : cluster.cluster_id)}
            type="button"
          >
            <span className="legend-dot" style={{ backgroundColor: cluster.color }} />
            {cluster.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("clustering");
  const [theme, setTheme] = useState(() => {
    const stored = window.localStorage.getItem("patent-dashboard-theme");
    return THEMES.includes(stored) ? stored : "light";
  });
  const [activeClusterId, setActiveClusterId] = useState(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("patent-dashboard-theme", theme);
  }, [theme]);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/dashboard/clusters`);
        if (!response.ok) {
          throw new Error(`Request failed with ${response.status}`);
        }
        const payload = await readJsonSafely(response);
        if (active) {
          setData(payload);
          if (payload.clusters.length > 0) {
            setActiveClusterId(payload.clusters[0].cluster_id);
          }
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Failed to load dashboard data");
        }
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  const activeCluster = useMemo(() => {
    if (!data) return null;
    return data.clusters.find((cluster) => cluster.cluster_id === activeClusterId) ?? data.clusters[0] ?? null;
  }, [data, activeClusterId]);

  if (error) {
    return (
      <main className="app-shell">
        <section className="hero">
          <p className="eyebrow">Patent intelligence dashboard</p>
          <h1>Dashboard unavailable</h1>
          <p className="hero-copy">{error}</p>
        </section>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="app-shell">
        <section className="hero">
          <p className="eyebrow">Patent intelligence dashboard</p>
          <h1>Loading cluster map</h1>
          <p className="hero-copy">Fetching clustering metrics, UMAP points, and label summaries from the backend.</p>
        </section>
      </main>
    );
  }

  const metricCards = [
    ["Documents", data.summary.num_documents, "default"],
    ["Clusters", data.summary.num_clusters, "accent"],
    ["ARI", data.summary.ari, "success"],
    ["NMI", data.summary.nmi, "success"],
    ["Purity", data.summary.purity, "success"],
    ["Silhouette", data.summary.silhouette, "default"],
  ];

  return (
    <main className="app-shell">
      <section className="hero hero-grid">
        <div className="hero-copy-block">
          <p className="eyebrow">Patent intelligence dashboard</p>
          <h1>Five clean semantic clusters, now with a richer UI.</h1>
          <p className="hero-copy">
            The clustering now runs in the original embedding space, which separates AI/ML and 5G correctly. The UI
            supports light and dark themes, interactive cluster focus, and a cleaner inspection surface for the next
            steps: chat and novelty scoring.
          </p>
        </div>

        <div className="hero-controls">
          <ThemeToggle
            theme={theme}
            onToggle={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
          />
          <div className="hero-note hero-note-strong">
            <span>Current run</span>
            <strong>{data.metrics.method}</strong>
            <p>
              Raw HDBSCAN noise points: <strong>{data.metrics.raw_noise_points}</strong> · reassigned:{" "}
              <strong>{data.metrics.noise_reassigned}</strong>
            </p>
          </div>
        </div>
      </section>

      <section className="shell-toolbar">
        <TabBar activeTab={activeTab} onSelectTab={setActiveTab} />
      </section>

      {activeTab === "clustering" ? (
        <>
          <section className="metrics-grid">
            {metricCards.map(([label, value, tone]) => (
              <MetricCard key={label} label={label} value={value} tone={tone} />
            ))}
          </section>

          <section className="main-grid">
            <div className="panel wide-panel">
              <div className="panel-head panel-head-row">
                <div>
                  <p className="panel-kicker">Visualization</p>
                  <h2>UMAP cluster formation</h2>
                </div>
                <button className="ghost-button" onClick={() => setActiveClusterId(null)} type="button">
                  Show all clusters
                </button>
              </div>
              <ClusterPlot
                points={data.points}
                clusters={data.clusters}
                activeClusterId={activeClusterId}
                onSelectCluster={setActiveClusterId}
              />
            </div>

            <div className="panel side-panel">
              <div className="panel-head">
                <div>
                  <p className="panel-kicker">Interpretation</p>
                  <h2>Cluster labels</h2>
                </div>
              </div>
              <div className="cluster-list">
                {data.clusters.map((cluster) => (
                  <ClusterCard
                    key={cluster.cluster_id}
                    cluster={cluster}
                    isActive={activeClusterId === cluster.cluster_id}
                    onSelect={setActiveClusterId}
                  />
                ))}
              </div>
            </div>
          </section>

          <ClusterInsight cluster={activeCluster} summary={data.summary} metrics={data.metrics} />
        </>
      ) : null}

      {activeTab === "novelty" ? <NoveltyTab /> : null}
      {activeTab === "rag" ? <RagTab /> : null}
    </main>
  );
}
