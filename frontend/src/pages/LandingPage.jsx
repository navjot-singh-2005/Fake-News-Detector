import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import styles from './LandingPage.module.css'

const FEATURES = [
  { icon: '🔗', title: 'URL Indicators', desc: 'Review HTTPS, URL structure, and simple domain-name heuristics. These do not verify an article.' },
  { icon: '🤖', title: 'Trainable ML Classification', desc: 'Compare TF-IDF Logistic Regression, Linear SVM, and Multinomial Naive Bayes on your labeled dataset.' },
  { icon: '📰', title: 'Related Search', desc: 'Search for related coverage when a News API key is configured; search results are not fact checks.' },
  { icon: '📊', title: 'Model Evaluation', desc: 'Training generates a holdout evaluation report with precision, recall, F1, accuracy, and confusion matrices.' },
  { icon: '⚡', title: 'Saved Pipeline', desc: 'Load the selected trained pipeline for predictions without retraining on each request.' },
  { icon: '🔒', title: 'Local History', desc: 'Analysis history is stored in this browser. Article text is not added to the history.' },
]

export default function LandingPage() {
  const orbsRef = useRef(null)

  useEffect(() => {
    document.title = 'TruthLens – AI Fake News Detector'
  }, [])

  return (
    <div className={styles.page}>
      {/* Animated Background */}
      <div className={styles.bg} ref={orbsRef} aria-hidden="true">
        <div className={styles.orb1} />
        <div className={styles.orb2} />
        <div className={styles.orb3} />
        <div className={styles.grid} />
      </div>

      {/* ── Hero ── */}
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <div className={styles.badge}>
            <span className={styles.badgeDot} />
            Trainable ML · Text Classification · Educational
          </div>

          <h1 className={styles.heroTitle}>
            Detect Fake News
            <br />
            <span className="gradient-text">Instantly</span>
          </h1>

          <p className={styles.heroSubtitle}>
            Paste a news article or headline to get a classification from a model trained on labeled data.
            Add a URL to review basic URL indicators separately.
          </p>

          <div className={styles.heroCta}>
            <Link to="/dashboard" id="hero-cta" className="btn-primary" style={{ padding: '14px 36px', fontSize: '1rem' }}>
              🚀 Analyze News Text
            </Link>
            <a href="#features" className="btn-secondary" style={{ padding: '14px 28px', fontSize: '1rem' }}>
              See How It Works ↓
            </a>
          </div>

        </div>

        {/* Hero Card Preview */}
        <div className={styles.heroCard} aria-hidden="true">
          <div className={styles.cardHeader}>
            <div className={styles.cardDots}>
              <span style={{ background: '#ef4444' }} />
              <span style={{ background: '#f59e0b' }} />
              <span style={{ background: '#10b981' }} />
            </div>
            <span className={styles.cardTitle}>TruthLens ML Pipeline</span>
          </div>
          <div className={styles.cardUrl}>
            <span className={styles.urlProtocol}>INPUT</span>
            <span className={styles.urlPath}>News article text or headline</span>
          </div>
          <div className={styles.cardResult}>
            <div className={styles.resultBadgeReal}>MODEL</div>
            <div className={styles.resultScore}>
              <span>Training data</span>
              <span className={styles.resultPct}>Required</span>
            </div>
          </div>
          <div className={styles.cardMetrics}>
            <div className={styles.metric}><span>Vectorizer</span><span className={styles.metricOk}>TF-IDF</span></div>
            <div className={styles.metric}><span>Algorithms</span><span className={styles.metricOk}>3 models</span></div>
            <div className={styles.metric}><span>Selection</span><span className={styles.metricOk}>5-fold CV</span></div>
            <div className={styles.metric}><span>Model artifact</span><span className={styles.metricOk}>Train locally</span></div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className={styles.features}>
        <div className="container">
          <p className="section-label">Features</p>
          <h2 className={styles.sectionTitle}>Everything you need to <span className="gradient-text">fight misinformation</span></h2>
          <p className={styles.sectionSub}>A data-driven text classifier with separate, explicitly heuristic URL indicators.</p>

          <div className={styles.featureGrid}>
            {FEATURES.map((f, i) => (
              <div key={i} className={`${styles.featureCard} glass-card`} style={{ animationDelay: `${i * 0.08}s` }}>
                <div className={styles.featureIcon}>{f.icon}</div>
                <h3 className={styles.featureTitle}>{f.title}</h3>
                <p className={styles.featureDesc}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section className={styles.howItWorks}>
        <div className="container">
          <p className="section-label">Process</p>
          <h2 className={styles.sectionTitle}>How <span className="gradient-text">TruthLens</span> works</h2>
          <div className={styles.steps}>
            {[
              { n: '01', title: 'Paste article text', desc: 'Provide the article or headline; the URL is optional.' },
              { n: '02', title: 'Train a model', desc: 'Use a labeled CSV dataset to compare three TF-IDF classifiers.' },
              { n: '03', title: 'Classify text', desc: 'The saved pipeline classifies the supplied text as REAL or FAKE.' },
              { n: '04', title: 'Verify independently', desc: 'Treat the result as a model estimate, not proof, and consult reliable sources.' },
            ].map((s, i) => (
              <div key={i} className={styles.step}>
                <div className={styles.stepNum}>{s.n}</div>
                <h3 className={styles.stepTitle}>{s.title}</h3>
                <p className={styles.stepDesc}>{s.desc}</p>
                {i < 3 && <div className={styles.stepArrow}>→</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ── */}
      <section className={styles.ctaBanner}>
        <div className={styles.ctaInner}>
          <h2 className={styles.ctaTitle}>Stop spreading misinformation.</h2>
          <p className={styles.ctaSub}>Model predictions are not definitive proof. Verify important claims independently.</p>
          <Link to="/dashboard" className="btn-primary" style={{ padding: '15px 40px', fontSize: '1.05rem' }}>
            Start Verifying Free →
          </Link>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <p>© {new Date().getFullYear()} TruthLens. News classification for research and education.</p>
        <p className={styles.footerSub}>For educational purposes. Always consult multiple sources.</p>
      </footer>
    </div>
  )
}
