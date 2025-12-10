import { useState, useEffect } from 'react';
import type { OperationLog } from '../types';
import { useApi } from '../hooks/useApi';

interface OperationsLogPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export function OperationsLogPanel({ isOpen, onClose }: OperationsLogPanelProps) {
  const [logs, setLogs] = useState<OperationLog[]>([]);
  const [expandedLogs, setExpandedLogs] = useState<Set<number>>(new Set());
  const { apiCall } = useApi();

  useEffect(() => {
    if (isOpen) {
      loadLogs();
      // Refresh logs every 5 seconds while panel is open
      const interval = setInterval(loadLogs, 5000);
      return () => clearInterval(interval);
    }
  }, [isOpen]);

  const loadLogs = async () => {
    try {
      const response = await apiCall<{ logs: OperationLog[] }>('/operation-logs?limit=50');
      setLogs(response.logs || []);
    } catch (error) {
      console.error('Error loading operation logs:', error);
    }
  };

  const toggleExpanded = (id: number) => {
    const newExpanded = new Set(expandedLogs);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedLogs(newExpanded);
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  const getOperationColor = (opType: string) => {
    switch (opType) {
      case 'START': return '#4CAF50';
      case 'STOP': return '#FF9800';
      case 'CANCEL': return '#F44336';
      case 'DELETE': return '#9E9E9E';
      case 'CREATE': return '#2196F3';
      default: return '#666';
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        right: 0,
        width: '500px',
        height: '100vh',
        backgroundColor: '#fff',
        boxShadow: '-2px 0 8px rgba(0,0,0,0.15)',
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '16px 20px',
          borderBottom: '1px solid #ddd',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: '#f5f5f5',
        }}
      >
        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Operations Log</h2>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            padding: '0 8px',
            color: '#666',
          }}
          title="Close"
        >
          ✕
        </button>
      </div>

      {/* Log entries */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
        {logs.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#999', padding: '40px 20px' }}>
            No operations yet
          </div>
        ) : (
          logs.map(log => {
            const isExpanded = expandedLogs.has(log.id);
            const hasOutput = log.output && log.output.trim().length > 0;

            return (
              <div
                key={log.id}
                style={{
                  marginBottom: '12px',
                  padding: '12px',
                  backgroundColor: log.success ? '#f9f9f9' : '#ffebee',
                  borderRadius: '6px',
                  border: `1px solid ${log.success ? '#e0e0e0' : '#ffcdd2'}`,
                }}
              >
                {/* Operation header */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 600,
                      backgroundColor: getOperationColor(log.operation_type),
                      color: 'white',
                    }}
                  >
                    {log.operation_type}
                  </span>
                  <span style={{ fontSize: '13px', color: '#666', fontFamily: 'monospace' }}>
                    {log.experiment_name || 'Unknown'}
                  </span>
                  <span style={{ marginLeft: 'auto', fontSize: '11px', color: '#999' }}>
                    {formatTimestamp(log.timestamp)}
                  </span>
                </div>

                {/* Error message if failed */}
                {!log.success && log.error_message && (
                  <div style={{ fontSize: '12px', color: '#c62828', marginBottom: '8px' }}>
                    Error: {log.error_message}
                  </div>
                )}

                {/* Output toggle */}
                {hasOutput && (
                  <>
                    <button
                      onClick={() => toggleExpanded(log.id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: '#2196F3',
                        fontSize: '12px',
                        cursor: 'pointer',
                        padding: '4px 0',
                        textDecoration: 'underline',
                      }}
                    >
                      {isExpanded ? '▼ Hide output' : '▶ Show output'}
                    </button>

                    {isExpanded && (
                      <pre
                        style={{
                          marginTop: '8px',
                          padding: '8px',
                          backgroundColor: '#f5f5f5',
                          borderRadius: '4px',
                          fontSize: '11px',
                          fontFamily: 'monospace',
                          overflow: 'auto',
                          maxHeight: '200px',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        {log.output}
                      </pre>
                    )}
                  </>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
