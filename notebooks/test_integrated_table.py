#!/usr/bin/env python3
"""
Example of how to integrate the RunStore HTML table with refresh functionality.
This demonstrates how to use the table in a Jupyter notebook with working refresh.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_store import RunStore


def create_integrated_table():
    """Create a RunStore table with integrated refresh functionality."""

    # Create RunStore instance
    rs = RunStore()

    # Generate a stable table ID
    table_id = "runstore_integrated_demo"

    # Generate the HTML table
    table_html = rs.to_html_table(table_id=table_id)

    # Create integration JavaScript that connects refresh buttons to RunStore
    integration_script = f"""
    <script>
    // Integration script for RunStore refresh functionality
    (function() {{
        // Store reference to RunStore instance
        window._runstore_instance = {id(rs)};
        
        // Refresh all runs callback
        window.{table_id}_refresh_all_callback = async function() {{
            console.log('Refreshing all runs...');
            
            // In a real Jupyter environment, you would call Python code here
            // For this demo, we'll simulate the refresh
            setTimeout(() => {{
                console.log('Refresh complete (simulated)');
                document.getElementById('{table_id}_loading').classList.remove('active');
                document.getElementById('{table_id}_refresh_all').disabled = false;
                document.querySelectorAll('.refresh-row-btn').forEach(btn => btn.disabled = false);
                
                // In real integration, you would re-render the table with fresh data
                alert('In a Jupyter notebook, this would call rs.refresh_all() and update the table.');
            }}, 1500);
        }};
        
        // Refresh single run callback
        window.{table_id}_refresh_run_callback = async function(runId) {{
            console.log('Refreshing run:', runId);
            
            // In a real Jupyter environment, you would call Python code here
            setTimeout(() => {{
                console.log('Refresh complete for:', runId);
                document.getElementById('{table_id}_loading').classList.remove('active');
                document.getElementById('{table_id}_refresh_all').disabled = false;
                document.querySelectorAll('.refresh-row-btn').forEach(btn => btn.disabled = false);
                
                // In real integration, you would re-render the table with fresh data
                alert('In a Jupyter notebook, this would call rs.refresh_run("' + runId + '") and update the table.');
            }}, 1000);
        }};
        
        // Track run callback
        window.{table_id}_track_run_callback = async function(runId) {{
            console.log('Adding run to RunStore:', runId);
            
            // In real integration, this would call rs.add_run(runId)
            setTimeout(() => {{
                console.log('Run added:', runId);
            }}, 500);
        }};
        
        console.log('RunStore integration script loaded');
    }})();
    </script>
    """

    # Combine table HTML with integration script
    full_html = table_html + integration_script

    return full_html


def jupyter_integration_example():
    """
    Example code for Jupyter notebook integration with real Python callbacks.
    Copy this into a Jupyter notebook cell:
    """
    example = '''
# In Jupyter Notebook:
from IPython.display import HTML, display, Javascript
from notebooks.run_store import RunStore

# Create RunStore instance and store it globally
rs = RunStore()
table_id = "runstore_live"

# Store RunStore instance for JavaScript access
get_ipython().user_ns['_runstore'] = rs

# Generate table
table_html = rs.to_html_table(table_id=table_id)

# Create proper integration with Python callbacks
integration_js = f"""
<script>
(function() {{
    // Refresh all runs
    window.{table_id}_refresh_all_callback = function() {{
        return new Promise((resolve, reject) => {{
            IPython.notebook.kernel.execute(
                '_runstore.refresh_all(); _table_html = _runstore.to_html_table(table_id="{table_id}")',
                {{
                    iopub: {{
                        output: function(msg) {{
                            if (msg.msg_type === 'error') {{
                                reject(msg.content);
                            }}
                        }}
                    }}
                }},
                {{
                    silent: false,
                    user_expressions: {{'table_html': '_table_html'}},
                    allow_stdin: false
                }}
            ).then(function(reply) {{
                const newHtml = reply.content.user_expressions.table_html.data['text/plain'];
                const cleanHtml = newHtml.slice(1, -1).replace(/\\\\n/g, '\\n').replace(/\\\\"/g, '"');
                document.getElementById('{table_id}_container').parentElement.innerHTML = cleanHtml;
                resolve();
            }});
        }});
    }};
    
    // Refresh single run
    window.{table_id}_refresh_run_callback = function(runId) {{
        return new Promise((resolve, reject) => {{
            IPython.notebook.kernel.execute(
                `_runstore.refresh_run("${{runId}}"); _table_html = _runstore.to_html_table(table_id="{table_id}")`,
                {{
                    iopub: {{
                        output: function(msg) {{
                            if (msg.msg_type === 'error') {{
                                reject(msg.content);
                            }}
                        }}
                    }}
                }},
                {{
                    silent: false,
                    user_expressions: {{'table_html': '_table_html'}},
                    allow_stdin: false
                }}
            ).then(function(reply) {{
                const newHtml = reply.content.user_expressions.table_html.data['text/plain'];
                const cleanHtml = newHtml.slice(1, -1).replace(/\\\\n/g, '\\n').replace(/\\\\"/g, '"');
                document.getElementById('{table_id}_container').parentElement.innerHTML = cleanHtml;
                resolve();
            }});
        }});
    }};
    
    // Track run
    window.{table_id}_track_run_callback = function(runId) {{
        return IPython.notebook.kernel.execute(`_runstore.add_run("${{runId}}")`);
    }};
}})();
</script>
"""

# Display the table with integration
display(HTML(table_html + integration_js))
'''
    return example


if __name__ == "__main__":
    # Generate demo HTML
    html = create_integrated_table()

    # Save to file
    with open("test_integrated_table.html", "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>RunStore Integrated Table Demo</title>
    <meta charset="utf-8">
</head>
<body style="margin: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <h1>RunStore Table with Integrated Refresh</h1>
    <p>This demo shows how the table works with integrated refresh callbacks.</p>
    <p>In a real Jupyter notebook, the refresh buttons would call Python RunStore methods.</p>
    <hr>
    {html}
    
    <hr>
    <h2>Jupyter Notebook Integration Example:</h2>
    <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto;">
{jupyter_integration_example()}
    </pre>
</body>
</html>""")

    print("Integrated table demo saved to test_integrated_table.html")
    print("\nTo test in Jupyter:")
    print("1. Copy the code from the 'Jupyter Notebook Integration Example' section")
    print("2. Paste into a Jupyter notebook cell")
    print("3. The refresh buttons will actually call rs.refresh_all() and rs.refresh_run()")
