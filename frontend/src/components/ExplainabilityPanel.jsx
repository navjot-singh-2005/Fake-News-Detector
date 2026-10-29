import styles from './ExplainabilityPanel.module.css'

export default function ExplainabilityPanel({ explanation, flags, topSignals, textMetrics }) {
  const hasRealSignals = topSignals?.real && topSignals.real.length > 0
  const hasFakeSignals = topSignals?.fake && topSignals.fake.length > 0
  const hasSignals = hasRealSignals || hasFakeSignals

  return (
    <div className={`${styles.card} glass-card`}>
      <div className={styles.header}>
        <span className={styles.icon}>🧠</span>
        <div>
          <p className="section-label">AI Explainability</p>
          <h2 className={styles.title}>Model Reasoning & Features</h2>
        </div>
      </div>

      <div className={styles.explanationBox}>
        <span className={styles.quoteIcon}>"</span>
        <p className={styles.explanation}>{explanation}</p>
      </div>

      {/* Model TF-IDF Feature Contributions */}
      {hasSignals && (
        <div className={styles.signalsSection}>
          <p className={styles.signalsTitle}>Model Indicative Keywords & N-Grams:</p>
          <div className={styles.signalsGrid}>
            {hasRealSignals && (
              <div className={styles.signalGroup}>
                <span className={styles.signalLabelReal}>✔ Terms Leaning Credible / Real</span>
                <div className={styles.tagList}>
                  {topSignals.real.map((word, i) => (
                    <span key={i} className={styles.tagReal}>
                      {word}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {hasFakeSignals && (
              <div className={styles.signalGroup}>
                <span className={styles.signalLabelFake}>✕ Terms Leaning Suspicious / Fake</span>
                <div className={styles.tagList}>
                  {topSignals.fake.map((word, i) => (
                    <span key={i} className={styles.tagFake}>
                      {word}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Linguistic & Sensationalism Metrics */}
      {textMetrics && (
        <div className={styles.metricsRow}>
          <div className={styles.metricItem}>
            <span className={styles.metricNum}>{textMetrics.word_count || textMetrics.wordCount || 0}</span>
            <span className={styles.metricLabel}>Words</span>
          </div>
          <div className={styles.metricItem}>
            <span
              className={styles.metricNum}
              style={{
                color:
                  (textMetrics.sensationalism_score || textMetrics.sensationalismScore || 0) > 40
                    ? 'var(--color-fake)'
                    : 'var(--color-real)',
              }}
            >
              {textMetrics.sensationalism_score || textMetrics.sensationalismScore || 0}%
            </span>
            <span className={styles.metricLabel}>Sensationalism Index</span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricNum}>
              {Math.round(((textMetrics.uppercase_ratio || textMetrics.uppercaseRatio || 0) * 100))}%
            </span>
            <span className={styles.metricLabel}>Capitalization</span>
          </div>
          <div className={styles.metricItem}>
            <span className={styles.metricNum}>
              {textMetrics.exclamation_count || textMetrics.exclamationCount || 0}
            </span>
            <span className={styles.metricLabel}>Exclamation Marks</span>
          </div>
        </div>
      )}

      {flags && flags.length > 0 && (
        <div className={styles.flagsSection}>
          <p className={styles.flagsTitle}>URL Risk Factors Detected:</p>
          <div className={styles.flagsList}>
            {flags.map((flag, i) => (
              <div key={i} className={styles.flagItem}>
                <span className={styles.flagDot} />
                {flag}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={styles.disclaimer}>
        <span>ℹ</span>
        <p>This analysis is AI-generated based on machine learning weights and heuristics. Always verify with multiple trusted sources before sharing news.</p>
      </div>
    </div>
  )
}
