import React from 'react';
import ReactDOM from 'react-dom';
import Plotly from 'plotly.js/dist/plotly';
import { Heatmap } from '../../../../../observatory/src/Heatmap';


function render({ model, el }) {
    // Create a simple test container first
    const container = document.createElement('div');
    container.style.cssText = `
        padding: 20px;
        border: 2px solid #007bff;
        border-radius: 8px;
        margin: 10px 0;
        background-color: #f8f9fa;
        font-family: Arial, sans-serif;
        min-height: 200px;
    `;

    const title = document.createElement('h3');
    title.style.cssText = `
        color: #007bff;
        margin-top: 0;
        margin-bottom: 20px;
    `;
    title.textContent = '📊 Interactive Policy Heatmap';

    const statusDiv = document.createElement('div');
    statusDiv.style.cssText = `
        padding: 10px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin-bottom: 15px;
    `;

    // Check if we have data
    const data = model.get('heatmap_data');
    if (!data || !data.cells || Object.keys(data.cells).length === 0) {
        statusDiv.innerHTML = '<strong>Status:</strong> No data available. Use widget.set_data() to load data.';
    } else {
        const policyCount = Object.keys(data.cells).length;
        const evalCount = data.evalNames ? data.evalNames.length : 0;
        statusDiv.innerHTML = `<strong>Status:</strong> Loaded ${policyCount} policies and ${evalCount} evaluations`;
    }

    const plotContainer = document.createElement('div');
    plotContainer.style.cssText = `
        width: 100%;
        height: 400px;
        border: 1px solid #ddd;
        background-color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 15px;
    `;

    // Simple test content or Plotly plot
    if (!data || !data.cells || Object.keys(data.cells).length === 0) {
        plotContainer.innerHTML = '<div style="color: #666; text-align: center;">Waiting for data...</div>';
    } else {
        // Try to load Plotly and create the real heatmap
        renderHeatmap(plotContainer, model, data);
    }

    const controlsContainer = document.createElement('div');
    controlsContainer.style.cssText = `
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    `;

    const numPoliciesLabel = document.createElement('label');
    numPoliciesLabel.textContent = 'Policies to show: ';
    numPoliciesLabel.style.fontWeight = 'bold';

    const numPoliciesInput = document.createElement('input');
    numPoliciesInput.type = 'number';
    numPoliciesInput.min = '1';
    numPoliciesInput.max = '100';
    numPoliciesInput.value = model.get('num_policies_to_show');
    numPoliciesInput.style.cssText = `
        width: 60px;
        padding: 5px;
        margin-left: 5px;
    `;

    const infoArea = document.createElement('div');
    infoArea.style.cssText = `
        padding: 10px;
        background-color: #e9ecef;
        border-radius: 4px;
        font-style: italic;
        min-height: 20px;
    `;
    infoArea.textContent = 'Widget loaded successfully!';

    container.appendChild(title);
    container.appendChild(statusDiv);
    container.appendChild(controlsContainer);
    controlsContainer.appendChild(numPoliciesLabel);
    controlsContainer.appendChild(numPoliciesInput);
    container.appendChild(plotContainer);
    container.appendChild(infoArea);

    // Event listeners
    numPoliciesInput.addEventListener('change', () => {
        const newValue = parseInt(numPoliciesInput.value);
        if (newValue > 0) {
            model.set('num_policies_to_show', newValue);
            model.save_changes();
            infoArea.textContent = `Policies to show updated to: ${newValue}`;
        }
    });

    // Listen for model changes
    model.on('change:heatmap_data', () => {
        console.log("Data changed, reloading...");
        const newData = model.get('heatmap_data');
        if (newData && newData.cells && Object.keys(newData.cells).length > 0) {
            renderHeatmap(plotContainer, model, newData);
            const policyCount = Object.keys(newData.cells).length;
            const evalCount = newData.evalNames ? newData.evalNames.length : 0;
            statusDiv.innerHTML = `<strong>Status:</strong> Loaded ${policyCount} policies and ${evalCount} evaluations`;
        }
    });

    model.on('change:selected_metric', () => {
        const metric = model.get('selected_metric');
        infoArea.textContent = `Metric updated to: ${metric}`;

        // Re-render the heatmap with the new metric
        const currentData = model.get('heatmap_data');
        if (currentData && currentData.cells && Object.keys(currentData.cells).length > 0) {
            renderHeatmap(plotContainer, model, currentData);
        }
    });

    // Append container to element
    el.appendChild(container);

    console.log("HeatmapWidget render completed successfully");
}


function renderHeatmap(plotContainer, model, data) {
    plotContainer.innerHTML = '<div style="color: #007bff; text-align: center;">Loading Plotly...</div>';
    try {
        createHeatmap(plotContainer, model, data);
    } catch (error) {
        plotContainer.innerHTML = `
            <div style="color: #dc3545; text-align: center; padding: 20px;">
                <strong>Error loading Plotly:</strong><br>
                ${error.message}<br>
                <small>Check browser console for details</small>
            </div>
        `;
    }
}


function createHeatmap(plotContainer, model, data) {
    try {
        function getShortName(evalName) {
            if (evalName === 'Overall') return evalName;
            return evalName.split('/').pop() || evalName;
        }

        function wandbUrl(policyName) {
            const entity = 'metta-research';
            const project = 'metta';
            let policyKey = policyName;
            if (policyName.includes(':v')) {
                policyKey = policyName.split(':v')[0];
            }
            return `https://wandb.ai/${entity}/${project}/runs/${policyKey}`;
        }

        const selectedMetric = model.get('selected_metric');
        const numPoliciesToShow = parseInt(model.get('num_policies_to_show'));

        // Process data similar to the React component
        const policies = Object.keys(data.cells);

        // Group eval names by category
        const evalsByCategory = new Map();
        data.evalNames.forEach(evalName => {
            const category = evalName.split('/')[0];
            if (!evalsByCategory.has(category)) {
                evalsByCategory.set(category, []);
            }
            evalsByCategory.get(category).push(evalName);
        });

        // Build x-labels: overall, then grouped by category
        const xLabels = ['overall'];
        const shortNameToEvalName = new Map();
        shortNameToEvalName.set('overall', 'overall');

        // Sort categories alphabetically, then envs within each category
        const sortedCategories = Array.from(evalsByCategory.keys()).sort();
        sortedCategories.forEach(category => {
            const envs = evalsByCategory.get(category).sort();
            envs.forEach(evalName => {
                const shortName = getShortName(evalName);
                xLabels.push(shortName);
                shortNameToEvalName.set(shortName, evalName);
            });
        });

        // Sort policies by average score (best at bottom for better visibility)
        const sortedPolicies = policies.sort((a, b) =>
            (data.policyAverageScores[a] || 0) - (data.policyAverageScores[b] || 0)
        );

        // Take the specified number of top policies (from the end of sorted list)
        const yLabels = sortedPolicies.slice(-numPoliciesToShow);

        const z = yLabels.map(policy => {
            const row = [data.policyAverageScores[policy] || 0]; // Overall score first

            // Add scores for each evaluation in order
            sortedCategories.forEach(category => {
                const envs = evalsByCategory.get(category).sort();
                envs.forEach(evalName => {
                    const cell = data.cells[policy] && data.cells[policy][evalName];
                    row.push(cell ? cell.value : 0);
                });
            });

            return row;
        });

        const yLabelTexts = yLabels.map(policy => {
            const url = wandbUrl(policy);
            return `<a href="${url}" target="_blank">${policy}</a>`;
        });

        const plotData = {
            z: z,
            x: xLabels,
            y: yLabels,
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: {
                title: {
                    text: selectedMetric || 'Value',
                },
            },
            hovertemplate: '<b>Policy:</b> %{y}<br><b>Evaluation:</b> %{x}<br><b>Value:</b> %{z}<extra></extra>',
        };

        const layout = {
            title: {
                text: `Policy Evaluation Report: ${selectedMetric || 'Heatmap'}`,
                font: {
                    size: 20,
                },
            },
            height: 400,
            width: Math.max(600, xLabels.length * 50),
            margin: { t: 50, b: 100, l: 150, r: 50 },
            xaxis: {
                tickangle: -45,
            },
            yaxis: {
                tickangle: 0,
                automargin: true,
                ticktext: yLabelTexts,
                tickvals: Array.from({ length: yLabels.length }, (_, i) => i),
                tickmode: 'array',
            },
        };

        plotContainer.innerHTML = ''; // Clear loading message

        console.log("Plotting heatmap...");
        Plotly.newPlot(plotContainer, [plotData], layout, {
            responsive: true,
            displayModeBar: true,
        }).then(() => {
            console.log("Done! Heatmap created successfully.");

            let lastHoveredCell: { policyUri: string; evalName: string } | null = null;

            // Add hover event listener
            plotContainer.on('plotly_hover', (eventData) => {
                if (!eventData.points || eventData.points.length === 0) return;

                const point = eventData.points[0];
                const shortName = point.x;
                const policyUri = point.y;
                const value = point.z;

                const evalName = shortNameToEvalName.get(shortName);
                lastHoveredCell = { policyUri, evalName };

                if (shortName !== 'overall') {
                    console.log("Setting selected cell...", { policyUri, evalName });
                    model.set('selected_cell', { policyUri, evalName });
                    model.save_changes();
                }
            });

            // Add double-click event listener
            plotContainer.on('plotly_doubleclick', () => {
                if (lastHoveredCell) {
                    if (lastHoveredCell.evalName === 'overall') {
                        alert("'overall' cells have no replays. Choose one of the other cells to watch a replay.");
                        return;
                    }
                    const cell = data.cells[lastHoveredCell.policyUri] && data.cells[lastHoveredCell.policyUri][lastHoveredCell.evalName];
                    if (cell && cell.replayUrl) {
                        const replayUrlPrefix = 'https://metta-ai.github.io/metta/?replayUrl=';
                        window.open(replayUrlPrefix + cell.replayUrl, '_blank');

                        console.log("Replay opened...", { policyUri: lastHoveredCell.policyUri, evalName: cell.evalName, replayUrl: cell.replayUrl });
                        model.set('replay_opened', {
                            policyUri: lastHoveredCell.policyUri,
                            evalName: lastHoveredCell.evalName,
                            replayUrl: cell.replayUrl
                        });
                        model.save_changes();
                    }
                }
            });
        }).catch(error => {
            console.error("Error creating Plotly plot:", error);
            plotContainer.innerHTML = `<div style="color: #dc3545; text-align: center;">Error creating plot: ${error.message}</div>`;
        });

    } catch (error) {
        console.error("Error in createHeatmap:", error);
        plotContainer.innerHTML = `<div style="color: #dc3545; text-align: center;">Error: ${error.message}</div>`;
    }
}


export default { render };
