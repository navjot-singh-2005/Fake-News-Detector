import styles from './HistoryPanel.module.css'

export default function HistoryPanel({ history, onSelect, onClear, onClose }) {
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h3>🕒 Analysis History</h3>
        <button className={styles.closeBtn} onClick={onClose} title="Close">✕</button>
      </div>

      {history.length === 0 ? (
        <div className={styles.empty}>
          <p>No history yet. Classify news text to get started.</p>
        </div>
      ) : (
        <>
          <div className={styles.list}>
            {history.map((item, index) => {
              const date = new Date(item.time).toLocaleDateString()
              return (
                <div key={index} className={styles.item} onClick={() => onSelect(item)}>
                  <div className={styles.itemHeader}>
                    <span className={`${styles.badge} ${item.label === 'FAKE' ? styles.fake : item.label === 'REAL' ? styles.real : styles.suspicious}`}>
                      {item.label}
                    </span>
                    <span className={styles.date}>{date}</span>
                  </div>
                  <div className={styles.url}>{item.url || 'News text classification'}</div>
                  {item.modelName && <div className={styles.url}>Model: {item.modelName}</div>}
                  {Number.isFinite(item.score) && (
                    <div className={styles.url}>Model probability: {item.score}%</div>
                  )}
                </div>
              )
            })}
          </div>
          <button className={styles.clearBtn} onClick={onClear}>Clear History</button>
        </>
      )}
    </div>
  )
}
