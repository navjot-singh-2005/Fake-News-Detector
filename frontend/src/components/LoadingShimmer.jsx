import styles from './LoadingShimmer.module.css'

export default function LoadingShimmer() {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.title} />
        <div className={styles.subtitle} />
      </div>
      <div className={styles.grid}>
        <div className={`${styles.card} ${styles.cardLarge}`} />
        <div className={styles.card} />
        <div className={styles.card} />
      </div>
      <div className={styles.bottomRow}>
        <div className={styles.card} />
        <div className={styles.card} />
      </div>
    </div>
  )
}
