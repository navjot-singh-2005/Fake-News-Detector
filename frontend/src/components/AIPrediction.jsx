import { useEffect, useRef } from 'react'
import styles from './AIPrediction.module.css'

const LABEL_CONFIG = {
  REAL:       { color: 'var(--color-real)',       icon: '✔', bg: 'var(--color-real-dim)',       glow: 'var(--glow-real)' },
  FAKE:       { color: 'var(--color-fake)',       icon: '✕', bg: 'var(--color-fake-dim)',       glow: 'var(--glow-fake)' },
  SUSPICIOUS: { color: 'var(--color-suspicious)', icon: '⚠', bg: 'var(--color-suspicious-dim)', glow: 'var(--glow-suspicious)' },
}

function CircularMeter({ value, color }) {
  const radius = 54
  const circ = 2 * Math.PI * radius
  const dashOffset = circ - (circ * value) / 100
  const circleRef = useRef(null)

  useEffect(() => {
    if (circleRef.current) {
      circleRef.current.style.strokeDashoffset = circ
      const timer = setTimeout(() => {
        if (circleRef.current) circleRef.current.style.strokeDashoffset = dashOffset
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [value, dashOffset, circ])

  return (
    <svg width="130" height="130" viewBox="0 0 130 130" className={styles.meter}>
      <circle cx="65" cy="65" r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
      <circle
        ref={circleRef}
        cx="65" cy="65" r={radius}
        fill="none"
        stroke={color}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={circ}
        strokeDashoffset={circ}
        transform="rotate(-90 65 65)"
        style={{
          transition: 'stroke-dashoffset 1.4s cubic-bezier(0.4,0,0.2,1)',
          filter: `drop-shadow(0 0 6px ${color})`,
        }}
      />
      <text x="65" y="60" textAnchor="middle" fill={color} fontSize="22" fontWeight="900" fontFamily="Space Grotesk, sans-serif">
        {value}%
      </text>
      <text x="65" y="76" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9" fontFamily="Inter, sans-serif">
        Model probability
      </text>
    </svg>
  )
}

export default function AIPrediction({ mlResult, finalLabel }) {
  const cfg = LABEL_CONFIG[finalLabel] || LABEL_CONFIG.SUSPICIOUS
  const { color, icon, bg, glow } = cfg
  const probabilities = mlResult.probabilities
  const modelFactors = Object.entries(probabilities || {})
    .map(([label, probability]) => ({ label, score: Math.round(probability * 100) }))

  return (
    <div className={`${styles.card} glass-card`}>
      <div className={styles.header}>
        <span className={styles.icon}>🤖</span>
        <div>
          <p className="section-label">AI Model</p>
          <h2 className={styles.title}>ML Prediction</h2>
        </div>
      </div>

      <div className={styles.verdictArea} style={{ '--c': color, '--bg': bg, '--glow': glow }}>
        <div className={styles.verdictIcon} style={{ background: bg, border: `1px solid ${color}`, color, boxShadow: glow }}>
          {icon}
        </div>
        <div className={styles.verdictText}>
          <span className={styles.verdictLabel} style={{ color }}>{finalLabel}</span>
          <span className={styles.verdictSub}>Classification Result</span>
        </div>
      </div>

      {Number.isFinite(mlResult.confidence) ? (
        <div className={styles.meterWrapper}>
          <CircularMeter value={mlResult.confidence} color={color} />
        </div>
      ) : probabilities && Object.keys(probabilities).length > 0 ? (
        <p className={styles.verdictSub}>
          {Math.round(Math.max(...Object.values(probabilities)) * 100)}% model probability for this classification
        </p>
      ) : (
        <p className={styles.verdictSub}>This model does not provide probability estimates.</p>
      )}

      {modelFactors.length > 0 && (
        <div className={styles.breakdown}>
          {modelFactors.map(({ label, score }) => (
            <div key={label} className={styles.factor}>
              <span className={styles.factorLabel}>{label}</span>
              <div className={styles.factorBar}>
                <div
                  className={styles.factorFill}
                  style={{ '--w': `${score}%`, background: color }}
                />
              </div>
              <span className={styles.factorScore} style={{ color }}>{score}%</span>
            </div>
          ))}
        </div>
      )}

      <div className={styles.modelInfo}>
        <span className={styles.modelTag}>{mlResult.modelName}</span>
        <span className={styles.modelTag}>TF-IDF</span>
      </div>
      <p className={styles.verdictSub}>Model estimates are not definitive proof that an article is true or false.</p>
    </div>
  )
}
