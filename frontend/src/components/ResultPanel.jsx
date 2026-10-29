import { useState } from 'react'
import SourceAnalysis from './SourceAnalysis'
import AIPrediction from './AIPrediction'
import CrossVerification from './CrossVerification'
import ExplainabilityPanel from './ExplainabilityPanel'
import TrustChart from './TrustChart'
import styles from './ResultPanel.module.css'

export default function ResultPanel({ result, relatedNews, relatedNewsLoading, relatedNewsError, url }) {
  const [copied, setCopied] = useState(false)
  const [actionError, setActionError] = useState('')

  const handleCopy = async () => {
    setActionError('')
    const text = `TruthLens Analysis\n${url ? `URL: ${url}\n` : ''}Verdict: ${result.finalLabel}\nModel: ${result.mlResult.modelName}\n\n${result.explanation}`
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      setActionError('Clipboard access failed. Copy is unavailable in this browser context.')
      return
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleShare = async () => {
    setActionError('')
    if (navigator.share) {
      try {
        await navigator.share({ title: 'TruthLens Result', text: `The supplied news text was classified as ${result.finalLabel} by ${result.mlResult.modelName}. This is not definitive proof.` })
      } catch (error) {
        if (error.name !== 'AbortError') setActionError('Sharing failed. Try copying the result instead.')
      }
    } else {
      handleCopy()
    }
  }

  const labelColor = result.finalLabel === 'REAL' ? 'var(--color-real)' : result.finalLabel === 'FAKE' ? 'var(--color-fake)' : 'var(--color-suspicious)'

  return (
    <div className={styles.container} id="result-panel">
      {/* Result Header Bar */}
      <div className={styles.resultHeader} style={{ '--accent': labelColor }}>
        <div className={styles.resultSummary}>
          <div className={styles.verdictBadge} style={{ background: `${labelColor}22`, border: `1px solid ${labelColor}`, color: labelColor, boxShadow: `0 0 20px ${labelColor}40` }}>
            {result.finalLabel === 'REAL' ? '✔' : result.finalLabel === 'FAKE' ? '✕' : '⚠'} {result.finalLabel}
          </div>
          <div className={styles.summaryText}>
            {url && <span className={styles.urlChip} title={url}>{url.length > 60 ? url.slice(0, 60) + '…' : url}</span>}
            <span className={styles.scoreChip}>Model: <strong style={{ color: labelColor }}>{result.mlResult.modelName}</strong></span>
          </div>
        </div>
        <div className={styles.resultActions}>
          <button id="copy-result" className={styles.actionBtn} onClick={handleCopy} title="Copy result">
            {copied ? '✔ Copied!' : '📋 Copy'}
          </button>
          <button id="share-result" className={styles.actionBtn} onClick={handleShare} title="Share result">
            🔗 Share
          </button>
        </div>
        {actionError && <span role="status">{actionError}</span>}
      </div>

      {/* 3 Panel Grid */}
      <div className={styles.panelGrid}>
        <SourceAnalysis features={result.urlFeatures} />
        <AIPrediction mlResult={result.mlResult} finalLabel={result.finalLabel} />
        <CrossVerification
          articles={relatedNews}
          loading={relatedNewsLoading}
          error={relatedNewsError}
        />
      </div>

      {/* Bottom Row */}
      <div className={styles.bottomRow}>
        <ExplainabilityPanel
          explanation={result.explanation}
          flags={result.urlFeatures?.flags || []}
          topSignals={result.mlResult?.topSignals}
          textMetrics={result.mlResult?.textMetrics}
        />
        <TrustChart breakdown={result.credibilityBreakdown} />
      </div>
    </div>
  )
}
