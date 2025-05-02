"""
Map Viewer component for evaluation visualization.

This module provides a reusable Map Viewer component that can be
integrated with other visualization components like heatmaps.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# CSS styles                                                                  #
# --------------------------------------------------------------------------- #
MAP_VIEWER_CSS = """
.map-viewer {
    position: relative;
    width: 100%;
    margin-top: 20px;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background: #f9f9f9;
    min-height: 300px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.map-viewer-title {
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
    font-size: 18px;
}
.map-viewer-img {
    max-width: 100%;
    max-height: 350px;
    display: block;
    margin: 0 auto;
}
.map-viewer-placeholder {
    text-align: center;
    color: #666;
    padding: 50px 0;
    font-style: italic;
}
.map-viewer-controls {
    display: flex;
    justify-content: center;
    margin-top: 15px;
    gap: 10px;
}
.map-button {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    font-size: 14px;
}
.map-button svg {
    width: 14px;
    height: 14px;
}
.map-button.locked {
    background: #f0f0f0;
    border-color: #aaa;
}
.map-button:hover {
    background: #f0f0f0;
}
.map-button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
"""


# --------------------------------------------------------------------------- #
# Map Viewer component                                                        #
# --------------------------------------------------------------------------- #
def create_map_viewer_html(uid: str) -> str:
    """
    Create the HTML for a Map Viewer component.

    Args:
        uid: Unique identifier to namespace this component's elements

    Returns:
        HTML snippet for the Map Viewer component
    """
    map_id = f"{uid}_map"
    map_title_id = f"{uid}_map_title"
    map_img_id = f"{uid}_map_img"
    map_placeholder_id = f"{uid}_map_placeholder"
    lock_button_id = f"{uid}_lock_button"
    replay_button_id = f"{uid}_replay_button"

    return f"""
<!-- Map Viewer Panel -->
<div class="map-viewer" id="{map_id}">
  <div class="map-viewer-title" id="{map_title_id}">Map Viewer</div>
  <div class="map-viewer-placeholder" id="{map_placeholder_id}">
    Hover over an evaluation name or cell to see the environment map
  </div>
  <img class="map-viewer-img" id="{map_img_id}" alt="Environment map" style="display: none;">
  
  <div class="map-viewer-controls">
    <button id="{lock_button_id}" class="map-button" title="Lock current view (or click cell)">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
        <path fill-rule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 
        2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" 
        clip-rule="evenodd" />
      </svg>
      <span>Lock View</span>
    </button>
    <button id="{replay_button_id}" class="map-button disabled" title="Open replay in Mettascope">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
        <path fill-rule="evenodd" d="M4.25 5.5a.75.75 0 00-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 
        00.75-.75v-4a.75.75 0 011.5 0v4A2.25 2.25 0 0112.75 17h-8.5A2.25 2.25 0 012 14.75v-8.5A2.25 2.25 
        0 014.25 4h5a.75.75 0 010 1.5h-5z" clip-rule="evenodd" />
        <path fill-rule="evenodd" d="M6.194 12.753a.75.75 0 001.06.053L16.5 4.44v2.81a.75.75 0 001.5
          0v-4.5a.75.75 0 00-.75-.75h-4.5a.75.75 0 000 1.5h2.553l-9.056 8.194a.75.75 0 00-.053 1.06z" 
          clip-rule="evenodd" />
      </svg>
      <span>Open Replay</span>
    </button>
  </div>
</div>
"""


# --------------------------------------------------------------------------- #
# JavaScript functions                                                        #
# --------------------------------------------------------------------------- #
def get_map_viewer_js_functions(uid: str) -> str:
    """
    Get JavaScript functions for the Map Viewer component.

    Args:
        uid: Unique identifier matching the one used for create_map_viewer_html

    Returns:
        JavaScript functions as a string, to be included in a <script> tag
    """
    map_id = f"{uid}_map"
    map_title_id = f"{uid}_map_title"
    map_img_id = f"{uid}_map_img"
    map_placeholder_id = f"{uid}_map_placeholder"
    lock_button_id = f"{uid}_lock_button"
    replay_button_id = f"{uid}_replay_button"

    return f"""
const mapPanel = document.getElementById("{map_id}");
const mapTitle = document.getElementById("{map_title_id}");
const mapImg = document.getElementById("{map_img_id}");
const mapPlaceholder = document.getElementById("{map_placeholder_id}");
const lockButton = document.getElementById("{lock_button_id}");
const replayButton = document.getElementById("{replay_button_id}");

let currentDisplayedEvalName = null;
let isMouseOverMap = false;
let isViewLocked = false;
let currentReplayUrl = null;

// Track when mouse enters/leaves the map viewer
mapPanel.addEventListener('mouseenter', function() {{
    isMouseOverMap = true;
}});

mapPanel.addEventListener('mouseleave', function() {{
    isMouseOverMap = false;
    if (!isViewLocked) {{
        setTimeout(() => {{
            if (!isMouseOverHeatmap && !isMouseOverMap) {{
                hideMap();
            }}
        }}, 100);
    }}
}});

// Lock/unlock button
lockButton.addEventListener('click', function() {{
    toggleLock();
}});

// Replay button
replayButton.addEventListener('click', function() {{
    if (currentReplayUrl) {{
        window.open(currentReplayUrl, '_blank');
    }}
}});

function toggleLock() {{
    isViewLocked = !isViewLocked;
    lockButton.classList.toggle('locked', isViewLocked);
    
    // Update button text
    const lockText = lockButton.querySelector('span');
    lockText.textContent = isViewLocked ? 'Unlock View' : 'Lock View';
}}

function showMap(name, replayUrl = null) {{
    if (isViewLocked) return;
    
    if (name.toLowerCase() === "overall") return;
    
    // Store the currently displayed evaluation name
    currentDisplayedEvalName = name;
    
    // Update replay button status
    if (replayUrl) {{
        currentReplayUrl = replayUrl;
        replayButton.classList.remove('disabled');
    }} else {{
        replayButton.classList.add('disabled');
        currentReplayUrl = null;
    }}
    
    const url = imgs[name] || "";
    
    // Show only the eval name in the title (ignore the policy part)
    mapTitle.textContent = name;
    
    if (url) {{
        mapImg.src = url;
        mapImg.style.display = "block";
        mapPlaceholder.style.display = "none";
    }} else {{
        mapImg.style.display = "none";
        mapPlaceholder.textContent = "No map available for " + name;
        mapPlaceholder.style.display = "block";
    }}
}}

function hideMap() {{
    // Only hide if we're not locked and not hovering over the map panel itself
    if (isViewLocked || isMouseOverMap) {{
        return;
    }}
    
    // Reset the current displayed evaluation name
    currentDisplayedEvalName = null;
    currentReplayUrl = null;
    
    // Update the replay button
    replayButton.classList.add('disabled');
    
    mapImg.style.display = "none";
    mapPlaceholder.textContent = "Hover over an evaluation name or cell to see the environment map";
    mapPlaceholder.style.display = "block";
}}
"""
