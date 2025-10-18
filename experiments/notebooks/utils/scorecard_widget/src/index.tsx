import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Scorecard } from './Scorecard';


interface ScorecardWidgetProps {
    model: {
        get: (key: string) => any;
        on: (event: string, callback: (...args: any[]) => void) => void;
        off: (event: string, callback: (...args: any[]) => void) => void;
        set: (key: string, value: any) => void;
        save_changes: () => void;
    };
}


function ScorecardWidget({ model }: ScorecardWidgetProps) {
    const [scorecardData, setScorecardData] = useState(null);
    const [selectedMetric, setSelectedMetric] = useState('');
    const [numPoliciesToShow, setNumPoliciesToShow] = useState(10);
    const [availableMetrics, setAvailableMetrics] = useState<string[]>([]);

    // Initialize state from model
    useEffect(() => {
        setScorecardData(model.get('scorecard_data'));
        setSelectedMetric(model.get('selected_metric'));
        setNumPoliciesToShow(model.get('num_policies_to_show'));
        setAvailableMetrics(model.get('available_metrics') ?? []);
    }, [model]);

    // Listen for model changes
    useEffect(() => {
        const handleDataChange = () => {
            console.log("Scorecard data changed, updating...");
            setScorecardData(model.get('scorecard_data'));
            setAvailableMetrics(model.get('available_metrics') ?? []);
        };

        const handleMetricChange = () => {
            console.log("Selected metric changed, updating...");
            setSelectedMetric(model.get('selected_metric'));
        };

        const handleNumPoliciesChange = () => {
            console.log("Num policies to show changed, updating...");
            setNumPoliciesToShow(model.get('num_policies_to_show'));
        };

        const handleAvailableMetricsChange = () => {
            console.log("Available metrics changed, updating...");
            setAvailableMetrics(model.get('available_metrics') ?? []);
        };

        model.on('change:scorecard_data', handleDataChange);
        model.on('change:selected_metric', handleMetricChange);
        model.on('change:num_policies_to_show', handleNumPoliciesChange);
        model.on('change:available_metrics', handleAvailableMetricsChange);

        return () => {
            model.off('change:scorecard_data', handleDataChange);
            model.off('change:selected_metric', handleMetricChange);
            model.off('change:num_policies_to_show', handleNumPoliciesChange);
            model.off('change:available_metrics', handleAvailableMetricsChange);
        };
    }, [model]);

    const setSelectedCell = (cell: { policyUri: string; evalName: string }) => {
        console.log("Setting selected cell:", cell);
        model.set('selected_cell', cell);
        model.save_changes();
    };

    const openReplayUrl = (policyName: string, evalName: string) => {
        console.log("Opening replay for:", { policyName, evalName });

        const cell = scorecardData?.cells[policyName]?.[evalName];
        if (!cell?.replayUrl) {
            console.warn("No replay URL found for cell:", { policyName, evalName });
            return;
        }

        const replayUrlPrefix = 'https://metta-ai.github.io/metta/?replayUrl=';
        window.open(replayUrlPrefix + cell.replayUrl, '_blank');

        // Signal back to Python that replay was opened
        model.set('replay_opened', {
            policyUri: policyName,
            evalName: evalName,
            replayUrl: cell.replayUrl
        });
        model.save_changes();
    };

    const handleMetricSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const newMetric = event.target.value;
        const currentMetric = selectedMetric || (availableMetrics[0] ?? '');
        if (!newMetric || newMetric === currentMetric) {
            return;
        }
        setSelectedMetric(newMetric);
        model.set('selected_metric', newMetric);
        model.save_changes();
    };

    // Transform data to use selected metric
    const transformedData = React.useMemo(() => {
        if (!scorecardData || !scorecardData.cells) return scorecardData;

        const metricKey = selectedMetric || scorecardData.availableMetrics?.[0] || availableMetrics[0] || '';
        const transformedCells: any = {};
        const transformedPolicyAverages: any = {};

        // Transform cells to use selected metric value
        Object.keys(scorecardData.cells).forEach(policyName => {
            transformedCells[policyName] = {};
            const policy = scorecardData.cells[policyName];

            Object.keys(policy).forEach(evalName => {
                const cell = policy[evalName];
                let value = 0;

                // Check if cell has metrics object or single value
                if (cell.metrics && typeof cell.metrics === 'object') {
                    if (metricKey && metricKey in cell.metrics) {
                        value = cell.metrics[metricKey] ?? 0;
                    } else {
                        const metricValues = Object.values(cell.metrics).filter(
                            (metricValue): metricValue is number => typeof metricValue === 'number'
                        );
                        if (metricValues.length > 0) {
                            value = metricValues[0];
                        } else if (typeof cell.value === 'number') {
                            value = cell.value;
                        } else if (cell.value != null) {
                            const parsed = Number(cell.value);
                            value = Number.isFinite(parsed) ? parsed : 0;
                        } else {
                            value = 0;
                        }
                    }
                } else {
                    // Fallback to single value (backwards compatibility)
                    if (typeof cell.value === 'number') {
                        value = cell.value;
                    } else if (cell.value != null) {
                        const parsed = Number(cell.value);
                        value = Number.isFinite(parsed) ? parsed : 0;
                    } else {
                        value = 0;
                    }
                }

                transformedCells[policyName][evalName] = {
                    ...cell,
                    value: value
                };
            });

            // Calculate policy average for selected metric
            const evalNames = Object.keys(policy);
            const total = evalNames.reduce((sum, evalName) => {
                const cell = policy[evalName];
                let value = 0;
                if (cell.metrics && typeof cell.metrics === 'object') {
                    if (metricKey && metricKey in cell.metrics) {
                        value = cell.metrics[metricKey] ?? 0;
                    } else {
                        const metricValues = Object.values(cell.metrics).filter(
                            (metricValue): metricValue is number => typeof metricValue === 'number'
                        );
                        if (metricValues.length > 0) {
                            value = metricValues[0];
                        } else if (typeof cell.value === 'number') {
                            value = cell.value;
                        } else if (cell.value != null) {
                            const parsed = Number(cell.value);
                            value = Number.isFinite(parsed) ? parsed : 0;
                        } else {
                            value = 0;
                        }
                    }
                } else if (typeof cell.value === 'number') {
                    value = cell.value;
                } else if (cell.value != null) {
                    const parsed = Number(cell.value);
                    value = Number.isFinite(parsed) ? parsed : 0;
                } else {
                    value = 0;
                }
                return sum + value;
            }, 0);
            transformedPolicyAverages[policyName] = evalNames.length > 0 ? total / evalNames.length : 0;
        });

        const evalAverageScores: Record<string, number> = {};
        const evalMaxScores: Record<string, number> = {};
        const policyNames = Object.keys(transformedCells);
        const evals = scorecardData.evalNames ?? [];

        evals.forEach(evalName => {
            const values = policyNames
                .map(policyName => {
                    const cell = transformedCells[policyName]?.[evalName];
                    const cellValue = cell?.value;
                    if (typeof cellValue === 'number') {
                        return cellValue;
                    }
                    if (typeof cellValue === 'string') {
                        const parsed = Number(cellValue);
                        return Number.isFinite(parsed) ? parsed : null;
                    }
                    return null;
                })
                .filter((value): value is number => typeof value === 'number');

            if (values.length > 0) {
                const sum = values.reduce((acc, value) => acc + value, 0);
                evalAverageScores[evalName] = sum / values.length;
                evalMaxScores[evalName] = Math.max(...values);
            } else {
                evalAverageScores[evalName] = 0;
                evalMaxScores[evalName] = 0;
            }
        });

        return {
            ...scorecardData,
            cells: transformedCells,
            policyAverageScores: transformedPolicyAverages,
            evalAverageScores,
            evalMaxScores
        };
    }, [scorecardData, selectedMetric, availableMetrics]);

    if (!scorecardData || !scorecardData.cells || Object.keys(scorecardData.cells).length === 0) {
        return (
            <div style={{
                padding: '20px',
                border: '2px solid #007bff',
                borderRadius: '8px',
                margin: '10px 0',
                backgroundColor: '#f8f9fa',
                fontFamily: 'Arial, sans-serif',
                textAlign: 'center'
            }}>
                <h3 style={{ color: '#007bff', marginTop: 0 }}>📊 Interactive Policy Scorecard</h3>
                <p style={{ color: '#666' }}>No data available. Use <code>widget.set_data()</code> to load data.</p>
            </div>
        );
    }

    // const policyCount = Object.keys(scorecardData.cells).length;
    // const evalCount = scorecardData.evalNames ? scorecardData.evalNames.length : 0;

    const effectiveMetric = selectedMetric || (availableMetrics[0] ?? '');

    return (
        <div style={{
            padding: '0',
            border: '2px solid #007bff',
            borderRadius: '8px',
            margin: '10px auto',
            backgroundColor: '#f8f9fa',
            fontFamily: 'Arial, sans-serif',
            maxWidth: '1100px',
            width: 'fit-content',
            display: 'block'
        }}>
            {availableMetrics.length > 0 && (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px 16px',
                    borderBottom: '1px solid #d9e3f0',
                    backgroundColor: '#fff',
                    borderTopLeftRadius: '6px',
                    borderTopRightRadius: '6px'
                }}>
                    <span style={{ fontSize: '14px', fontWeight: 600, color: '#007bff' }}>Metric</span>
                    <select
                        value={effectiveMetric}
                        onChange={handleMetricSelect}
                        disabled={availableMetrics.length <= 1}
                        style={{
                            padding: '6px 10px',
                            borderRadius: '6px',
                            border: '1px solid #c5d3e6',
                            backgroundColor: '#fdfdfd',
                            fontSize: '14px',
                            color: '#333'
                        }}
                    >
                        {availableMetrics.map(metric => (
                            <option key={metric} value={metric}>
                                {metric}
                            </option>
                        ))}
                    </select>
                </div>
            )}
            <Scorecard
                data={transformedData}
                selectedMetric={effectiveMetric}
                setSelectedCell={setSelectedCell}
                openReplayUrl={openReplayUrl}
                numPoliciesToShow={numPoliciesToShow}
            />
        </div>
    );
}


// Store roots by element to avoid recreating them
const rootMap = new WeakMap();

function render({ model, el }: { model: any; el: HTMLElement }) {
    console.log("ScorecardWidget render called");

    // Get or create root for this element
    let root = rootMap.get(el);
    if (!root) {
        el.innerHTML = '';
        root = createRoot(el);
        rootMap.set(el, root);
    }

    root.render(<ScorecardWidget model={model} />);
    console.log("ScorecardWidget render completed successfully");
}


export default { render };
