import styles from './CrossVerification.module.css'

export default function CrossVerification({ articles, loading, error }) {
  return (
    <div className={`${styles.card} glass-card`}>
      <div className={styles.header}>
        <span className={styles.icon}>📰</span>
        <div>
          <p className="section-label">Cross-Verification</p>
          <h2 className={styles.title}>Related Sources</h2>
        </div>
      </div>

      {!articles || articles.length === 0 ? (
        <div className={styles.empty}>
          <span>{error ? 'ℹ' : loading ? '🔄' : '📰'}</span>
          <p>{error || (loading ? 'Searching for related articles…' : 'No related articles are available.')}</p>
        </div>
      ) : (
        <div className={styles.articleList}>
          {articles.map((a, i) => {
            return (
              <div key={i} className={styles.article} style={{ animationDelay: `${i * 0.1}s` }}>
                <div className={styles.articleHeader}>
                  <span className={styles.source}>Search result</span>
                  <span className={styles.source}>{a.source}</span>
                </div>
                <p className={styles.articleTitle}>{a.title}</p>
                <div className={styles.articleFooter}>
                  {a.url && (
                    <a href={a.url} target="_blank" rel="noopener noreferrer" className={styles.articleLink}>↗</a>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      <p className={styles.legend}>Related search results do not establish whether a claim has been confirmed or contradicted.</p>
    </div>
  )
}
