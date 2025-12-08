import type { HealthData, HealthBackend } from '../types';
import './HealthStatus.css';

interface HealthStatusProps {
  health: HealthData | null;
  backendStaleness: number;
}

function formatBackend(name: string, backend: HealthBackend | null, intervalSeconds: number) {
  if (!backend || backend.staleness_seconds === null || backend.staleness_seconds === undefined) {
    return (
      <span key={name} className="staleness-pill staleness-unknown">
        {name}: —
      </span>
    );
  }

  const staleness = backend.staleness_seconds;
  const threshold1 = intervalSeconds * 1.2;
  const threshold2 = intervalSeconds * 3.0;

  let className = 'staleness-ok';
  if (staleness >= threshold2) {
    className = 'staleness-bad';
  } else if (staleness >= threshold1) {
    className = 'staleness-warn';
  }

  return (
    <span key={name} className={`staleness-pill ${className}`}>
      {name}: {staleness.toFixed(1)}s
    </span>
  );
}

export function HealthStatus({ health, backendStaleness }: HealthStatusProps) {
  if (!health) {
    return (
      <div className="health-status">
        <span className="status-indicator error"></span>
        <span>Loading...</span>
      </div>
    );
  }

  if (health.status !== 'ok') {
    return (
      <div className="health-status">
        <span className="status-indicator error"></span>
        <span>Error</span>
      </div>
    );
  }

  const backendObj: HealthBackend = { staleness_seconds: backendStaleness };

  return (
    <div className="health-status">
      {formatBackend('backend', backendObj, 5)}
      {formatBackend('skypilot', health.skypilot, 30)}
      {formatBackend('s3', health.s3, 60)}
      {formatBackend('obs', health.observatory, 60)}
      <span className="health-summary">
        {health.num_experiments} experiments | {health.num_running_jobs} active jobs
      </span>
    </div>
  );
}
