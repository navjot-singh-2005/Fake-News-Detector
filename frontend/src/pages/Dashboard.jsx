import { useState, useEffect, useRef } from 'react'
import { analyzeUrl } from '../api/analyzeService'
import { getRelatedNews } from '../api/newsService'
import ResultPanel from '../components/ResultPanel'
import LoadingShimmer from '../components/LoadingShimmer'
import HistoryPanel from '../components/HistoryPanel'
import styles from './Dashboard.module.css'

const MAX_ARTICLE_LENGTH = 100_000

export default function Dashboard() {
  const [url, setUrl] = useState('')
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [relatedNews, setRelatedNews] = useState([])
  const [relatedNewsLoading, setRelatedNewsLoading] = useState(false)
  const [relatedNewsError, setRelatedNewsError] = useState('')
  const [error, setError] = useState('')
  const [showHistory, setShowHistory] = useState(false)
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('tl-history') || '[]') } catch { return [] }
  })
  const [showText, setShowText] = useState(true)
  const requestIdRef = useRef(0)
  const resultRef = useRef(null)

  useEffect(() => { document.title = 'Dashboard – TruthLens' }, [])

  useEffect(() => {
    if (result) resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [result])

  const saveToHistory = (entry) => {
    const updated = [entry, ...history].slice(0, 20)
    setHistory(updated)
    localStorage.setItem('tl-history', JSON.stringify(updated))
  }

  const handleAnalyze = async () => {
    const hasText = Boolean(text.trim())
    const hasUrl = Boolean(url.trim())
    if (!hasText && !hasUrl) {
      setError('Please enter a news article URL or paste text/headline to classify.')
      return
    }
    if (text.length > MAX_ARTICLE_LENGTH) {
      setError(`Article text must be no longer than ${MAX_ARTICLE_LENGTH.toLocaleString()} characters.`)
      return
    }
    setError('')
    setResult(null)
    setRelatedNews([])
    setRelatedNewsLoading(false)
    setRelatedNewsError('')
    const requestId = ++requestIdRef.current
    setLoading(true)
    try {
      const data = await analyzeUrl(url.trim(), text.trim())
      setResult(data)
      if (data.extractedText && !text.trim()) {
        setText(data.extractedText)
        setShowText(true)
      }
      let query = ''
      if (data.extractedTitle) {
        query = data.extractedTitle.split(/\s+/).slice(0, 8).join(' ')
      } else if (url.trim()) {
        try {
          query = new URL(url.trim()).hostname.split('.')[0]
        } catch {
          query = ''
        }
      } else {
        query = text.trim().split(/\s+/).slice(0, 8).join(' ')
      }
      saveToHistory({
        url: url.trim(),
        label: data.finalLabel,
        score: data.finalScore,
        modelName: data.mlResult?.modelName || 'TF-IDF Classifier',
        time: new Date().toISOString(),
      })
      if (query) {
        setRelatedNewsLoading(true)
        getRelatedNews(query)
          .then(newsData => {
            if (requestId === requestIdRef.current) {
              setRelatedNews(newsData.articles || [])
              setRelatedNewsError('')
            }
          })
          .catch(newsError => {
            if (requestId === requestIdRef.current) {
              setRelatedNewsError(newsError.message)
            }
          })
          .finally(() => {
            if (requestId === requestIdRef.current) setRelatedNewsLoading(false)
          })
      }
    } catch (e) {
      setError(e.message || 'Analysis failed. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) handleAnalyze()
  }

  const handleClear = () => {
    requestIdRef.current += 1
    setUrl(''); setText(''); setResult(null); setError(''); setRelatedNews([]); setRelatedNewsError('')
    setRelatedNewsLoading(false)
  }

  const handleHistorySelect = (item) => {
    requestIdRef.current += 1
    setUrl(item.url || '')
    setText('')
    setShowText(true)
    setError('History keeps prediction details but not article text. Paste the text again to run another prediction.')
    setResult(null)
    setShowHistory(false)
  }

  return (
    <div className={styles.page}>
      {/* Bg */}
      <div className={styles.bg} aria-hidden="true">
        <div className={styles.orb1} />
        <div className={styles.orb2} />
      </div>

      <div className={styles.layout}>
        {/* ── Sidebar History ── */}
        <aside className={`${styles.sidebar} ${showHistory ? styles.sidebarOpen : ''}`}>
          <HistoryPanel
            history={history}
            onSelect={handleHistorySelect}
            onClear={() => { setHistory([]); localStorage.removeItem('tl-history') }}
            onClose={() => setShowHistory(false)}
          />
        </aside>

        {/* ── Main Content ── */}
        <main className={styles.main}>
          {/* Header */}
          <div className={styles.pageHeader}>
            <div>
              <p className="section-label">Analysis Dashboard</p>
              <h1 className={styles.pageTitle}>News <span className="gradient-text">Classification</span></h1>
              <p className={styles.pageSubtitle}>Paste news text or a headline for a model-based classification. A URL is optional.</p>
            </div>
            <button
              id="history-toggle"
              className={styles.historyBtn}
              onClick={() => setShowHistory(s => !s)}
              title="View analysis history"
            >
              🕒 History <span className={styles.historyCount}>{history.length}</span>
            </button>
          </div>

          {/* ── Input Card ── */}
          <div className={`${styles.inputCard} glass-card`}>
            <div className={styles.inputRow}>
              <div className={styles.urlInputWrapper}>
                <span className={styles.urlIcon}>🔗</span>
                <input
                  id="url-input"
                  type="url"
                  className={`input-field ${styles.urlInput}`}
                  placeholder="https://example.com/article (optional)"
                  value={url}
                  onChange={e => { setUrl(e.target.value); setError('') }}
                  onKeyDown={handleKeyDown}
                  disabled={loading}
                  aria-label="Optional news article URL"
                />
                {url && (
                  <button className={styles.clearBtn} onClick={handleClear} title="Clear" disabled={loading}>✕</button>
                )}
              </div>

              <button
                id="verify-btn"
                className="btn-primary"
                onClick={handleAnalyze}
                disabled={loading || (!text.trim() && !url.trim()) || text.length > MAX_ARTICLE_LENGTH}
                style={{ padding: '14px 28px', whiteSpace: 'nowrap' }}
              >
                {loading ? <span className={styles.spinner} /> : '🔍'} {loading ? 'Classifying...' : 'Classify News'}
              </button>
            </div>

            <div className={styles.textToggle}>
              <button
                className={styles.toggleBtn}
                onClick={() => setShowText(s => !s)}
                id="text-toggle"
              >
                {showText ? '▲' : '▼'} {text.trim() ? 'Edit article text' : 'Article text or headline (optional if URL provided)'}
              </button>
            </div>

            {showText && (
              <textarea
                id="article-text"
                className={`input-field ${styles.textarea}`}
                placeholder="Paste the article text or headline to classify (or enter a URL above to auto-extract)…"
                value={text}
                onChange={e => {
                  setText(e.target.value)
                  setError(e.target.value.length > MAX_ARTICLE_LENGTH
                    ? `Article text must be no longer than ${MAX_ARTICLE_LENGTH.toLocaleString()} characters.`
                    : '')
                }}
                disabled={loading}
                rows={8}
                aria-label="News article text"
              />
            )}
            {showText && (
              <p className={styles.pageSubtitle} aria-live="polite">
                {text.length.toLocaleString()} / {MAX_ARTICLE_LENGTH.toLocaleString()} characters
              </p>
            )}

            <div className={styles.inputMeta}>
              <div className={styles.signals}>
                <span className={styles.signal}>🔒 HTTPS Check</span>
                <span className={styles.signal}>📄 Article text</span>
                <span className={styles.signal}>🤖 AI Model</span>
                <span className={styles.signal}>📰 Related search</span>
                <span className={styles.signal}>🔗 URL heuristics</span>
              </div>
              <button
                id="check-credibility-btn"
                className="btn-secondary"
                onClick={handleAnalyze}
                disabled={loading || (!text.trim() && !url.trim()) || text.length > MAX_ARTICLE_LENGTH}
                style={{ padding: '9px 20px', fontSize: '0.85rem' }}
              >
                {loading ? 'Classifying...' : 'Classify Article'}
              </button>
            </div>

            {error && (
              <div className={styles.error} role="alert">
                <span>⚠</span> {error}
              </div>
            )}
          </div>

          {/* ── Results ── */}
          {loading && <LoadingShimmer />}
          {!loading && result && (
            <div ref={resultRef}>
              <ResultPanel
                result={result}
                relatedNews={relatedNews}
                relatedNewsLoading={relatedNewsLoading}
                relatedNewsError={relatedNewsError}
                url={url}
              />
            </div>
          )}

          {/* ── Empty State ── */}
          {!loading && !result && (
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}>🔍</div>
              <h3>Ready to classify</h3>
              <p>Paste article text or a headline above. You can optionally add its URL for separate URL indicators.</p>
              <div className={styles.exampleUrls}>
                <p className={styles.exampleLabel}>A model prediction is not definitive proof that a claim is true or false.</p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
