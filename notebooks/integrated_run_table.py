"""
Integrated RunStore table with working refresh functionality.
This uses IPython's display system for better integration.
"""

from IPython.display import HTML, Javascript, display
from run_store import get_runstore
import json


def create_integrated_table():
    """Create an integrated RunStore table with working buttons."""
    rs = get_runstore()
    
    # Generate the base table
    table_html = rs.to_html_table()
    
    # Create JavaScript that uses IPython's kernel directly
    js_code = """
    <script>
    // Wait for the table to be rendered
    setTimeout(function() {
        // Find all refresh buttons and track button
        const tableId = document.querySelector('[id^="runstore_"]').id;
        
        // Override the callbacks with IPython kernel execution
        if (window[tableId + '_track_run_callback']) {
            const originalTrack = window[tableId + '_track_run_callback'];
            window[tableId + '_track_run_callback'] = function(runId) {
                console.log('Tracking run via IPython:', runId);
                
                // Add to UI immediately
                originalTrack(runId);
                
                // Execute Python via IPython
                IPython.notebook.kernel.execute(
                    'from run_store import get_runstore; ' +
                    'rs = get_runstore(); ' +
                    'rs.add_run("' + runId + '"); ' +
                    'print("Added run: ' + runId + '")',
                    {
                        iopub: {
                            output: function(msg) {
                                if (msg.msg_type === 'stream') {
                                    console.log('Python output:', msg.content.text);
                                } else if (msg.msg_type === 'error') {
                                    console.error('Python error:', msg.content);
                                }
                            }
                        }
                    }
                );
                
                // Re-run the cell after a delay
                setTimeout(function() {
                    IPython.notebook.execute_cell();
                }, 1500);
            };
        }
        
        if (window[tableId + '_refresh_all_callback']) {
            window[tableId + '_refresh_all_callback'] = function() {
                console.log('Refreshing all via IPython');
                
                // Show loading
                document.getElementById(tableId + '_loading').classList.add('active');
                
                // Execute Python
                IPython.notebook.kernel.execute(
                    'from run_store import get_runstore; ' +
                    'rs = get_runstore(); ' +
                    'rs.refresh_all(); ' +
                    'print("Refreshed all runs")',
                    {
                        iopub: {
                            output: function(msg) {
                                if (msg.msg_type === 'stream') {
                                    console.log('Python output:', msg.content.text);
                                }
                            }
                        }
                    }
                );
                
                // Re-run the cell
                setTimeout(function() {
                    IPython.notebook.execute_cell();
                }, 1500);
            };
        }
        
        if (window[tableId + '_refresh_run_callback']) {
            window[tableId + '_refresh_run_callback'] = function(runId) {
                console.log('Refreshing run via IPython:', runId);
                
                // Show loading
                document.getElementById(tableId + '_loading').classList.add('active');
                
                // Execute Python
                IPython.notebook.kernel.execute(
                    'from run_store import get_runstore; ' +
                    'rs = get_runstore(); ' +
                    'rs.refresh_run("' + runId + '"); ' +
                    'print("Refreshed run: ' + runId + '")',
                    {
                        iopub: {
                            output: function(msg) {
                                if (msg.msg_type === 'stream') {
                                    console.log('Python output:', msg.content.text);
                                }
                            }
                        }
                    }
                );
                
                // Re-run the cell
                setTimeout(function() {
                    IPython.notebook.execute_cell();
                }, 1500);
            };
        }
        
        console.log('Integrated RunStore table initialized');
    }, 100);
    </script>
    """
    
    return HTML(table_html + js_code)


def show_runstore_table():
    """Display the RunStore table with integrated functionality."""
    display(create_integrated_table())
    

# Export for easy use
__all__ = ['show_runstore_table', 'create_integrated_table']