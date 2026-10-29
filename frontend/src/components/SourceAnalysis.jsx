import { useEffect, useState } from 'react'
import styles from './SourceAnalysis.module.css'

function AnimatedBar({ value, color }) {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const t = setTimeout(() => setWidth(value), 100)
    return () => clearTimeout(t)
  }, [value])
  return (
    <div className={styles.barTrack}>
      <div className={styles.barFill} style={{ width: `${width}%`, background: color, transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)' }} />
    </div>
  )
}

export default function SourceAnalysis({ features }) {
  if (!features) return null

  const { domain, isHttps, isTrusted, urlLength, pathDepth, hasSuspiciousTLD, suspiciousKeywords, trustScore, flags } = features

  const scoreColor = trustScore >= 70 ? 'var(--color-real)' : trustScore >= 40 ? 'var(--color-suspicious)' : 'var(--color-fake)'

  const rows = [
    { label: 'HTTPS Secure', value: isHttps ? '✔ Yes' : '✕ No', ok: isHttps },
    { label: 'Recognized domain list', value: isTrusted ? 'Hostname match (not verification)' : 'No match', info: true },
    { label: 'TLD Risk', value: hasSuspiciousTLD ? '⚠ Suspicious' : '✔ Normal', ok: !hasSuspiciousTLD },
    { label: 'URL Length', value: `${urlLength} chars`, ok: urlLength < 100 },
    { label: 'Path Depth', value: pathDepth, ok: pathDepth <= 6 },
  ]

  return (
    <div className={`${styles.card} glass-card`}>
      <div className={styles.header}>
        <span className={styles.icon}>🌐</span>
        <div>
          <p className="section-label">Source Analysis</p>
          <h2 className={styles.title}>URL Signals</h2>
        </div>
      </div>

      {/* Trust Score */}
      <div className={styles.scoreSection}>
        <div className={styles.scoreCircle} style={{ '--score-color': scoreColor, '--score': `${trustScore}%` }}>
          <span className={styles.scoreNum} style={{ color: scoreColor }}>{trustScore}</span>
          <span className={styles.scoreLabel}>/ 100</span>
        </div>
        <div className={styles.scoreRight}>
          <p className={styles.domain}>{domain}</p>
          <p className={styles.scoreSub}>Heuristic indicator (not a credibility score)</p>
          <AnimatedBar value={trustScore} color={scoreColor} />
        </div>
      </div>

      {/* Metrics Table */}
      <div className={styles.metricsTable}>
        {rows.map((r, i) => (
          <div key={i} className={styles.metricRow}>
            <span className={styles.metricLabel}>{r.label}</span>
            <span className={`${styles.metricValue} ${r.info ? styles.info : r.ok ? styles.ok : styles.warn}`}>{r.value}</span>
          </div>
        ))}
      </div>

      {/* Flags */}
      {flags && flags.length > 0 && (
        <div className={styles.flags}>
          <p className={styles.flagsTitle}>⚠ Risk Signals</p>
          {flags.map((f, i) => (
            <div key={i} className={styles.flag}>{f}</div>
          ))}
        </div>
      )}

      {suspiciousKeywords && suspiciousKeywords.length > 0 && (
        <div className={styles.keywords}>
          {suspiciousKeywords.map(k => (
            <span key={k} className={styles.keyword}>{k}</span>
          ))}
        </div>
      )}
    </div>
  )
}
