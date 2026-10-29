import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'
import { Radar } from 'react-chartjs-2'
import styles from './TrustChart.module.css'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

export default function TrustChart({ breakdown }) {
  if (!breakdown) return null

  const metrics = [
    ['Domain URL heuristics', breakdown.domainTrust],
    ['HTTPS', breakdown.httpsScore],
    ['Model probability estimate', breakdown.contentScore],
    ['URL cleanliness heuristic', breakdown.urlClean],
  ].filter(([, value]) => Number.isFinite(value))
  const labels = metrics.map(([label]) => label)
  const values = metrics.map(([, value]) => value)

  const avgScore = Math.round(values.reduce((a, b) => a + b, 0) / values.length)

  const data = {
    labels,
    datasets: [
      {
        label: 'Credibility Score',
        data: values,
        backgroundColor: 'rgba(34, 211, 238, 0.12)',
        borderColor: 'rgba(34, 211, 238, 0.8)',
        pointBackgroundColor: 'rgba(34, 211, 238, 1)',
        pointBorderColor: 'rgba(15,23,42,0.8)',
        pointBorderWidth: 2,
        pointRadius: 5,
        borderWidth: 2,
        fill: true,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(15,23,42,0.9)',
        borderColor: 'rgba(34,211,238,0.3)',
        borderWidth: 1,
        titleColor: '#22d3ee',
        bodyColor: '#94a3b8',
        callbacks: {
          label: ctx => ` ${ctx.raw}%`,
        },
      },
    },
    scales: {
      r: {
        min: 0,
        max: 100,
        ticks: {
          display: false,
          stepSize: 25,
        },
        grid: {
          color: 'rgba(255,255,255,0.06)',
        },
        angleLines: {
          color: 'rgba(255,255,255,0.06)',
        },
        pointLabels: {
          color: 'rgba(148,163,184,0.9)',
          font: { size: 11, family: 'Inter, sans-serif', weight: '500' },
        },
      },
    },
    animation: {
      duration: 1200,
      easing: 'easeInOutQuart',
    },
  }

  return (
    <div className={`${styles.card} glass-card`}>
      <div className={styles.header}>
        <span className={styles.icon}>📊</span>
        <div>
          <p className="section-label">Credibility Radar</p>
          <h2 className={styles.title}>Signal Breakdown</h2>
        </div>
      </div>

      <div className={styles.avgScore}>
        <span className={styles.avgNum} style={{ color: avgScore >= 70 ? 'var(--color-real)' : avgScore >= 40 ? 'var(--color-suspicious)' : 'var(--color-fake)' }}>
          {avgScore}%
        </span>
        <span className={styles.avgLabel}>Avg Credibility</span>
      </div>

      <div className={styles.chartWrapper}>
        <Radar data={data} options={options} />
      </div>

      <div className={styles.breakdown}>
        {labels.map((label, i) => (
          <div key={label} className={styles.breakdownItem}>
            <span className={styles.breakdownLabel}>{label}</span>
            <div className={styles.breakdownBar}>
              <div
                className={styles.breakdownFill}
                style={{ '--bw': `${values[i]}%` }}
              />
            </div>
            <span className={styles.breakdownVal}>{values[i]}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}
