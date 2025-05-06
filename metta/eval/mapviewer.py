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

    return f"""
<!-- Map Viewer Panel -->
<div class="map-viewer" id="{map_id}">
  <div class="map-viewer-title" id="{map_title_id}">Map Viewer</div>
  <div class="map-viewer-placeholder" id="{map_placeholder_id}">
    Hover over an evaluation name or cell to see the environment map
  </div>
  <img class="map-viewer-img" id="{map_img_id}" alt="Environment map" style="display: none;">
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

    return f"""
const mapPanel = document.getElementById("{map_id}");
const mapTitle = document.getElementById("{map_title_id}");
const mapImg = document.getElementById("{map_img_id}");
const mapPlaceholder = document.getElementById("{map_placeholder_id}");
let currentDisplayedEvalName = null;
let isMouseOverMap = false;

// Track when mouse enters/leaves the map viewer
mapPanel.addEventListener('mouseenter', function() {{
    isMouseOverMap = true;
}});

mapPanel.addEventListener('mouseleave', function() {{
    isMouseOverMap = false;
}});

function showMap(name, title) {{
  if (name.toLowerCase() === "overall") return;
  
  // Store the currently displayed evaluation name
  currentDisplayedEvalName = name;
  
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
  // Only hide if we're not hovering over the map panel itself
  if (isMouseOverMap) {{
    return;
  }}
  
  // Reset the current displayed evaluation name
  currentDisplayedEvalName = null;
  
  mapImg.style.display = "none";
  mapPlaceholder.textContent = "Hover over an evaluation name or cell to see the map";
  mapPlaceholder.style.display = "block";
}}
"""
