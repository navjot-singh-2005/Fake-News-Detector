import { Link, useLocation } from 'react-router-dom'
import styles from './Navbar.module.css'

export default function Navbar({ theme, toggleTheme }) {
  const location = useLocation()

  return (
    <nav className={styles.navbar}>
      <div className={styles.inner}>
        <Link to="/" className={styles.logo}>
          <span className={styles.logoIcon}>🔍</span>
          <span className={styles.logoText}>Truth<span className={styles.logoAccent}>Lens</span></span>
        </Link>

        <div className={styles.navLinks}>
          <Link to="/" className={`${styles.navLink} ${location.pathname === '/' ? styles.active : ''}`}>Home</Link>
          <Link to="/dashboard" className={`${styles.navLink} ${location.pathname === '/dashboard' ? styles.active : ''}`}>Dashboard</Link>
          <Link to="/#features" className={styles.navLink}>About</Link>
        </div>

        <div className={styles.navRight}>
          <button
            id="theme-toggle"
            className={styles.themeBtn}
            onClick={toggleTheme}
            aria-label="Toggle theme"
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <Link to="/dashboard" className="btn-primary" style={{ padding: '9px 20px', fontSize: '0.85rem' }}>
            Try Free →
          </Link>
        </div>
      </div>
    </nav>
  )
}
