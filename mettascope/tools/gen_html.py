import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import requests


def rgba(color: Dict[str, float], opacity=1) -> str:
    """
    Convert a Figma color object to CSS rgba color string.

    Args:
        color: A dictionary with r, g, b, a values in the range 0-1

    Returns:
        str: CSS rgba color string
    """
    r = round(color.get("r", 0) * 255)
    g = round(color.get("g", 0) * 255)
    b = round(color.get("b", 0) * 255)
    a = color.get("a", 1) * opacity

    return f"rgba({r}, {g}, {b}, {a:0.2f})"


def px(value: Any) -> str:
    """
    Convert a value to CSS pixel string with 2 decimal places.
    Removes trailing zeros and decimal point if the result is a whole number.

    Args:
        value: Numeric value

    Returns:
        str: Formatted CSS pixel value
    """
    # If its None its 0
    if value is None:
        return "0"

    # Convert to float first to handle any input type
    num = float(value)

    # Round to 2 decimal places
    rounded = round(num, 2)

    # If it's a whole number, convert to int to remove decimal
    if rounded == int(rounded):
        return f"{int(rounded)}px"

    # Format with exactly 2 decimal places
    return f"{rounded:.2f}px"


def sanitize_class_name(name: str) -> str:
    """
    Sanitize a class name to be valid CSS.

    Args:
        name: Raw class name from Figma

    Returns:
        str: Valid CSS class name
    """
    # Replace invalid characters with hyphens
    sanitized = name.replace("/", "-").replace(" ", "-").replace("_", "-")
    # Remove multiple consecutive hyphens
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")
    # Convert to lowercase for consistency
    return sanitized.lower()


def parse_name(name: str) -> tuple[str, str, list[str], str]:
    """Parse figma name into tag, id or class"""
    tags = ["input", "img", "textarea", "button", "a", "canvas", "iframe"]
    tag = "div"
    id = ""
    clss = []
    for part in name.split("."):
        if part in tags:
            tag = part
        elif part.startswith("#"):
            id = part[1:]
        else:
            # Use the sanitize function instead of just replacing spaces
            clss.append(sanitize_class_name(part))
    selector = ""
    if tag and tag not in ["div", "span"]:
        selector = tag
    if id:
        selector += "#" + id
    if len(clss) > 0:
        selector += "." + ".".join(clss)
    return (tag, id, clss, selector)


class DomNode:
    """A simple DOM node implementation for HTML generation."""

    # List of HTML5 self-closing tags
    SELF_CLOSING_TAGS = [
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    ]

    def __init__(self, tag_name: str, attributes: Optional[Dict[str, str]] = None, text: str = ""):
        """Initialize a DOM node."""
        self.tag_name = tag_name
        self.attributes = attributes or {}
        self.children = []
        self.text = text

    def append_child(self, child: "DomNode"):
        """Add a child node."""
        self.children.append(child)
        return child

    def set_attribute(self, name: str, value: str):
        """Set an attribute on the node."""
        self.attributes[name] = value
        return self

    def render(self, indent: int = 0) -> str:
        """Render the node to HTML string with proper indentation."""
        indent_str = "  " * indent
        html = []

        # Opening tag with attributes
        attr_str = ""
        if self.attributes:
            attr_str = " " + " ".join(f'{k}="{v}"' for k, v in self.attributes.items())

        # Handle different rendering cases
        is_self_closing = self.tag_name.lower() in self.SELF_CLOSING_TAGS and not self.children

        if is_self_closing:
            # Self-closing tag
            html.append(f"{indent_str}<{self.tag_name}{attr_str}>")
        else:
            # Normal tag
            html.append(f"{indent_str}<{self.tag_name}{attr_str}>")

            # Add text content if present
            if self.text:
                if self.children:
                    # If there are children, indent the text
                    html.append(f"{indent_str}  {self.text}")
                else:
                    # If no children, keep the text inline
                    html.append(self.text)

            # Add children
            for child in self.children:
                html.append(child.render(indent + 1))

            # Closing tag (with proper indentation if there are children)
            if self.children:
                html.append(f"{indent_str}</{self.tag_name}>")
            else:
                html.append(f"</{self.tag_name}>")

        return "\n".join(html) if self.children or self.tag_name in ["html", "head", "body"] else "".join(html)


class HtmlGenerator:
    """Class to handle the generation of HTML and CSS from Figma data."""

    def __init__(self, output_dir: str, extra_css: str, extra_js: str, data_dir: str, tmp_dir: str):
        """
        Initialize the HTML generator.

        Args:
            output_dir: Directory where generated files will be saved
            default_image_path: Path for image references
            default_font_family: Default font family for text elements
        """

        self.output_dir = output_dir
        self.extra_js = extra_js
        self.extra_css = extra_css
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir

    def gen_css_file(self, name: str, css_rules: List[Dict[str, Any]], url: str) -> str:
        """
        Generate a CSS file from a list of CSS rules.

        Args:
            name: Base name for the CSS file
            css_rules: List of CSS rules, each with 'selector' and 'styles'
            url: URL of the Figma file

        Returns:
            str: Path to the generated CSS file relative to output directory
        """
        # Create CSS content
        css_content = f"/* Generated CSS from Figma file: {url} by tools/gen_html.py */\n\n"

        # Add global styles
        css_content += """
*, *::before, *::after {
    margin: 0;
    padding: 0;
    border: none;
    box-sizing: border-box;
}
html, body {
    font-family: sans-serif;
}
"""

        # Add element-specific styles
        for rule in css_rules:
            selector = rule["selector"]
            styles = rule["styles"]
            if isinstance(styles, dict):
                css_content += f"\n{selector} {{\n"
                for prop, value in styles.items():
                    css_content += f"    {prop}: {value};\n"
                css_content += "}\n"

        # Write CSS to file
        css_file = f"{self.output_dir}/{name}.css"
        os.makedirs(os.path.dirname(css_file), exist_ok=True)
        with open(css_file, "w") as f:
            f.write(css_content)

        # Return relative path for linking
        return f"{name}.css"

    def gen_html_page(self, frame: Dict[str, Any], url: str) -> str:
        """
        Generate HTML for a frame from Figma.

        Args:
            frame: Frame data from Figma API
            url: URL of the Figma file

        Returns:
            str: Generated HTML content
        """

        # Extract frame name without .html extension
        name = frame["name"].replace(".html", "")

        # Create doctype
        doctype = "<!DOCTYPE html>"

        # Create comment
        comment = f"<!-- Generated from Figma file: {url} by tools/gen_html.py -->"

        # Create HTML document structure
        html_node = DomNode("html")
        html_node.set_attribute("lang", "en")
        head = html_node.append_child(DomNode("head"))
        body = html_node.append_child(DomNode("body"))

        # Add head elements
        head.append_child(DomNode("title", {}, name))
        head.append_child(DomNode("meta", {"charset": "UTF-8"}))
        head.append_child(DomNode("meta", {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}))

        # CSS rules for elements
        css_rules = []

        # Get background color from frame
        css = {}
        if "backgroundColor" in frame:
            css["background-color"] = rgba(frame["backgroundColor"])
        elif "fills" in frame and frame["fills"] and frame["fills"][0].get("type") == "SOLID":
            css["background-color"] = rgba(frame["fills"][0]["color"])
        self.compute_auto_layout(css, frame, None)
        css["min-width"] = "100vw"
        css["min-height"] = "100vh"
        css_rules.append({"selector": "body", "styles": css})

        # Process the frame children directly
        if "children" in frame:
            for child in frame["children"]:
                # Skip groups but process their children
                if child.get("type") == "GROUP":
                    if "children" in child:
                        for group_child in child["children"]:
                            child_element, child_css = self.process_element(group_child, frame)
                            body.append_child(child_element)
                            css_rules.extend(child_css)
                else:
                    child_element, child_css = self.process_element(child, frame)
                    body.append_child(child_element)
                    css_rules.extend(child_css)

        # Generate CSS file
        css_file = self.gen_css_file(name, css_rules, url)

        # Add CSS link to head
        head.append_child(DomNode("link", {"rel": "stylesheet", "href": css_file}))

        if self.extra_js:
            head.append_child(DomNode("script", {"type": "module", "src": self.extra_js}))

        if self.extra_css:
            head.append_child(DomNode("link", {"rel": "stylesheet", "href": self.extra_css}))

        # Generate HTML string
        html_content = doctype + "\n" + comment + "\n" + html_node.render() + "\n"

        # Write HTML to file
        output_file = f"{self.output_dir}/{name}.html"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(html_content)

        print(f"# {name}.html")
        return html_content

    def compute_constraints(
        self, css: Dict[str, Any], element: Dict[str, Any], parent: Optional[Dict[str, Any]] = None
    ) -> None:
        """Compute the constraints from a figma element."""
        # Get positioning data if available
        box = element["absoluteBoundingBox"]

        # Get constraints if available
        constraints = element.get("constraints", {})
        horizontal_constraint = constraints.get("horizontal", "LEFT")
        vertical_constraint = constraints.get("vertical", "TOP")

        # Get the box dimensions
        width = box.get("width", 0)
        height = box.get("height", 0)

        if parent and "absoluteBoundingBox" in parent:
            x = box.get("x", 0) - parent["absoluteBoundingBox"].get("x", 0)
            y = box.get("y", 0) - parent["absoluteBoundingBox"].get("y", 0)
            right = parent["absoluteBoundingBox"].get("width", 0) - (x + width)
            bottom = parent["absoluteBoundingBox"].get("height", 0) - (y + height)
            center_x = x + width / 2 - parent["absoluteBoundingBox"].get("width", 0) / 2
            center_y = y + height / 2 - parent["absoluteBoundingBox"].get("height", 0) / 2
        else:
            print("No parent for ", element["name"])
            x = box.get("x", 0)
            y = box.get("y", 0)
            right = 0
            bottom = 0
            center_x = 0
            center_y = 0

        translate_x = None
        translate_y = None

        # Width and height are always needed for proper sizing
        css["display"] = "block"
        css["width"] = px(width)
        css["height"] = px(height)

        # Apply horizontal constraints
        if horizontal_constraint == "LEFT":
            css["left"] = px(x)
        elif horizontal_constraint == "RIGHT":
            css["right"] = px(right)
        elif horizontal_constraint == "CENTER":
            # Center horizontally
            css["left"] = "50%"
            # Center offset
            translate_x = center_x - width / 2
        elif horizontal_constraint == "SCALE":
            # For SCALE, we want the element to scale with the parent
            # Both left and right distances must be maintained
            css["left"] = px(x)
            css["right"] = px(right)
            print("here right???")
            # Since we're using both left and right, we don't need width
            del css["width"]
        elif horizontal_constraint == "LEFT_RIGHT":
            css["left"] = px(x)
            css["right"] = px(right)
            del css["width"]

        # Apply vertical constraints
        if vertical_constraint == "TOP":
            css["top"] = px(y)
        elif vertical_constraint == "BOTTOM":
            css["bottom"] = px(bottom)
        elif vertical_constraint == "CENTER":
            # Center vertically
            css["top"] = "50%"
            # Center offset
            translate_y = center_y - height / 2
        elif vertical_constraint == "SCALE":
            # For SCALE, we want the element to scale with the parent
            # Both top and bottom distances must be maintained
            css["top"] = px(y)
            css["bottom"] = px(bottom)
            # Since we're using both top and bottom, we don't need height
            del css["height"]
        elif vertical_constraint == "TOP_BOTTOM":
            css["top"] = px(y)
            css["bottom"] = px(bottom)
            del css["height"]

        # Apply translation if they are needed (for center constraints)
        if translate_x is not None or translate_y is not None:
            css["transform"] = f"translate({px(translate_x)}, {px(translate_y)})"

        if "minWidth" in element:
            css["min-width"] = px(element["minWidth"])
        if "maxWidth" in element:
            css["max-width"] = px(element["maxWidth"])
        if "minHeight" in element:
            css["min-height"] = px(element["minHeight"])
        if "maxHeight" in element:
            css["max-height"] = px(element["maxHeight"])

        if "width" in css and ("minWidth" in element or "maxWidth" in element):
            del css["width"]
        if "height" in css and ("minHeight" in element or "maxHeight" in element):
            del css["height"]

    def compute_auto_layout(
        self, css: Dict[str, Any], element: Dict[str, Any], parent: Optional[Dict[str, Any]]
    ) -> None:
        """Compute CSS Flexbox properties from Figma Auto Layout settings."""

        if "layoutMode" in element:
            # Set display to flex for auto layout containers
            css["display"] = "flex"

            # Set flex direction based on layoutMode
            layout_mode = element.get("layoutMode")
            if layout_mode == "HORIZONTAL":
                css["flex-direction"] = "row"
            elif layout_mode == "VERTICAL":
                css["flex-direction"] = "column"

            # Handle flex wrap
            layout_wrap = element.get("layoutWrap")
            if layout_wrap == "WRAP":
                css["flex-wrap"] = "wrap"
            else:
                css["flex-wrap"] = "nowrap"

            # Set gap between items
            item_spacing = element.get("itemSpacing")
            if item_spacing is not None:
                css["gap"] = px(item_spacing)

            # Set padding
            padding_left = element.get("paddingLeft")
            padding_right = element.get("paddingRight")
            padding_top = element.get("paddingTop")
            padding_bottom = element.get("paddingBottom")

            # Only set padding if at least one value is provided
            if any(p is not None for p in [padding_left, padding_right, padding_top, padding_bottom]):
                padding_values = []
                for p in [padding_top, padding_right, padding_bottom, padding_left]:
                    padding_values.append(px(p) if p is not None else "0px")
                css["padding"] = " ".join(padding_values)

            # Handle alignment properties
            primary_axis_alignment = element.get("primaryAxisAlignItems")
            if primary_axis_alignment == "MIN":
                css["justify-content"] = "flex-start"
            elif primary_axis_alignment == "CENTER":
                css["justify-content"] = "center"
            elif primary_axis_alignment == "MAX":
                css["justify-content"] = "flex-end"
            elif primary_axis_alignment == "SPACE_BETWEEN":
                css["justify-content"] = "space-between"

            counter_axis_alignment = element.get("counterAxisAlignItems")
            if counter_axis_alignment == "MIN":
                css["align-items"] = "flex-start"
            elif counter_axis_alignment == "CENTER":
                css["align-items"] = "center"
            elif counter_axis_alignment == "MAX":
                css["align-items"] = "flex-end"

            # Handle sizing modes
            primary_axis_sizing = element.get("primaryAxisSizingMode")
            counter_axis_sizing = element.get("counterAxisSizingMode")

            if primary_axis_sizing == "FIXED" or primary_axis_sizing is None:
                # Fixed size is handled by the width/height already set
                pass
            elif primary_axis_sizing == "AUTO":
                if layout_mode == "HORIZONTAL":
                    css["width"] = "auto"
                else:
                    css["height"] = "auto"
            else:
                print("Unknown primary axis sizing mode:", primary_axis_sizing)

            if counter_axis_sizing == "FIXED" or counter_axis_sizing is None:
                # Fixed size is handled by the width/height already set
                pass
            elif counter_axis_sizing == "AUTO":
                if layout_mode == "HORIZONTAL":
                    css["height"] = "auto"
                else:
                    css["width"] = "auto"
            else:
                print("Unknown counter axis sizing mode:", counter_axis_sizing)

        if parent and "layoutMode" in parent:
            # Handle auto layout child properties
            layout_pos = element.get("layoutPositioning")
            if layout_pos == "ABSOLUTE":
                # Don't apply flex properties to absolutely positioned elements
                return

            layout_align = element.get("layoutAlign")
            layout_grow = element.get("layoutGrow", 0)

            if css["position"] == "absolute":
                # If the element is absolutely positioned, we need to remove the:
                # left, right, top, and bottom properties.
                # And set the position to relative.
                if "left" in css:
                    del css["left"]
                if "right" in css:
                    del css["right"]
                if "top" in css:
                    del css["top"]
                if "bottom" in css:
                    del css["bottom"]
                css["position"] = "relative"

            if layout_align == "STRETCH":
                css["align-self"] = "stretch"
            elif layout_align == "CENTER":
                css["align-self"] = "center"
            elif layout_align == "MIN":
                css["align-self"] = "flex-start"
            elif layout_align == "MAX":
                css["align-self"] = "flex-end"

            # Set flex grow
            if layout_grow is not None and layout_grow > 0:
                css["flex-grow"] = str(layout_grow)
                css["flex-shrink"] = "1"
                css["flex-basis"] = "auto"

        layout_sizing_horizontal = element.get("layoutSizingHorizontal")
        layout_sizing_vertical = element.get("layoutSizingVertical")

        if layout_sizing_horizontal == "FIXED" or layout_sizing_horizontal is None:
            # Fixed size is handled by the width/height already set
            pass
        elif layout_sizing_horizontal == "FILL":
            css["width"] = "100%"
        elif layout_sizing_horizontal == "HUG":
            css["width"] = "fit-content"
        else:
            print("Unknown layout sizing horizontal:", layout_sizing_horizontal)

        if layout_sizing_vertical == "FIXED" or layout_sizing_vertical is None:
            # Fixed size is handled by the width/height already set
            pass
        elif layout_sizing_vertical == "FILL":
            css["height"] = "100%"
        elif layout_sizing_vertical == "HUG":
            css["height"] = "fit-content"
        else:
            print("Unknown layout sizing vertical:", layout_sizing_vertical)

    def compute_text_properties(self, css: Dict[str, Any], element: Dict[str, Any]) -> None:
        """Computes the text properties"""
        if "fills" in element:
            for fill in element["fills"]:
                css["color"] = rgba(fill["color"], fill.get("opacity", 1.0))

        if "style" in element:
            style = element["style"]
            if "fontFamily" in style:
                css["font-family"] = style["fontFamily"]

            if "fontStyle" in style:
                # Figma uses "Regular" which maps to CSS "normal"
                css["font-style"] = "normal" if style["fontStyle"].lower() == "regular" else style["fontStyle"].lower()

            if "fontWeight" in style:
                css["font-weight"] = style["fontWeight"]

            if "fontSize" in style:
                css["font-size"] = px(style["fontSize"])

            if "textAlignHorizontal" in style:
                css["text-align"] = style["textAlignHorizontal"].lower()

            if "textAlignVertical" in style:
                # No exact match in CSS for vertical text alignment on blocks
                if style["textAlignVertical"].lower() == "top":
                    css["vertical-align"] = "top"
                elif style["textAlignVertical"].lower() == "center":
                    css["vertical-align"] = "middle"
                elif style["textAlignVertical"].lower() == "bottom":
                    css["vertical-align"] = "bottom"
                else:
                    print("Unknown vertical text alignment:", style["textAlignVertical"])

            if "letterSpacing" in style:
                css["letter-spacing"] = px(style["letterSpacing"])

            if "lineHeightPx" in style:
                css["line-height"] = px(round(style["lineHeightPx"], 2))

    def compute_rotation(self, css: Dict[str, Any], element: Dict[str, Any]) -> None:
        """Compute the rotation from a figma element."""
        if "rotation" in element and abs(element["rotation"]) > 0.01:
            css["transform-origin"] = "center"
            css["transform"] = f"rotate({element['rotation']}rad)"

    def compute_background(self, css: Dict[str, Any], element: Dict[str, Any]) -> None:
        """Compute the background from a figma element."""
        if "backgroundColor" in element:
            css["background-color"] = rgba(element["backgroundColor"])
        if "fills" in element:
            for fills in element["fills"]:
                if fills.get("type") == "SOLID":
                    css["background-color"] = rgba(fills["color"], fills.get("opacity", 1.0))

    def compute_border_radius(self, css: Dict[str, Any], element: Dict[str, Any]) -> None:
        """Compute the border radius from a figma element."""
        if "cornerRadius" in element and element["cornerRadius"] > 0:
            # Single radius for all corners
            css["border-radius"] = px(element["cornerRadius"])
        elif "rectangleCornerRadii" in element:
            # Different radius for each corner
            radii = element["rectangleCornerRadii"]
            if isinstance(radii, list) and len(radii) == 4 and any(r > 0 for r in radii):
                # Format: top-left, top-right, bottom-right, bottom-left
                css["border-radius"] = " ".join(px(r) for r in radii)

        if "clipsContent" in element and element["clipsContent"]:
            # Clip child elements to not go outside the bounds of the parent
            css["overflow"] = "hidden"

    def compute_stroke(self, css: Dict[str, Any], element: Dict[str, Any]) -> None:
        """Compute the stroke from a figma element."""
        if "strokes" in element and element["strokes"]:
            stroke = element["strokes"][0]
            if stroke.get("type") == "SOLID" and "color" in stroke:
                # Set border color
                css["border-color"] = rgba(stroke["color"])

                # Set border width if available
                if "strokeWeight" in element and element["strokeWeight"] > 0:
                    css["border-width"] = px(element["strokeWeight"])
                    # Default to solid border style
                    css["border-style"] = "solid"

                # Check stroke alignment and warn if not INSIDE
                if "strokeAlign" in element:
                    stroke_align = element.get("strokeAlign")
                    element_name = element.get("name", "")
                    element_id = element.get("id", "")
                    if stroke_align and stroke_align != "INSIDE":
                        print(
                            f"Warning: Element '{element_name}' (ID: {element_id}) uses {stroke_align} stroke"
                            + "alignment which is not supported in HTML/CSS. Using INSIDE stroke instead.",
                        )

        if "individualStrokeWeights" in element:
            weights = element["individualStrokeWeights"]
            css["border-top-width"] = px(weights["top"])
            css["border-right-width"] = px(weights["right"])
            css["border-bottom-width"] = px(weights["bottom"])
            css["border-left-width"] = px(weights["left"])

    def compute_effects(self, css: Dict[str, Any], element: Dict[str, Any]) -> None:
        """Computes the effects"""
        if "effects" in element:
            shadows = []
            for effect in element["effects"]:
                if not effect.get("visible", True):
                    continue  # skip invisible effects

                offset_x = effect["offset"]["x"]
                offset_y = effect["offset"]["y"]
                blur_radius = effect.get("radius", 0)
                css_color = rgba(effect["color"])

                if effect["type"] == "DROP_SHADOW":
                    shadows.append(f"{offset_x}px {offset_y}px {blur_radius}px {css_color}")
                elif effect["type"] == "INNER_SHADOW":
                    shadows.append(f"inset {offset_x}px {offset_y}px {blur_radius}px {css_color}")
                else:
                    print("Effect:", effect["type"], "not supported")

            if shadows:
                css["box-shadow"] = ", ".join(shadows)

    def process_element(self, element: Dict[str, Any], parent: Optional[Dict[str, Any]] = None) -> tuple:
        """Process a Figma element and convert it to HTML DOM node."""
        element_type = element.get("type")
        element_name = element.get("name", "")

        # Combine classes if they're different
        (tag, id, clss, selector) = parse_name(element_name)

        # CSS rules for this element and its children
        css_rules = []

        # Always set position to absolute for Figma elements
        css = {"position": "absolute"}

        # Compute constraints
        self.compute_constraints(css, element, parent)

        # Compute auto layout
        self.compute_auto_layout(css, element, parent)

        # Compute rotation
        self.compute_rotation(css, element)

        if element_type == "TEXT":
            self.compute_text_properties(css, element)
        else:
            # Get background color if available
            self.compute_background(css, element)

            # Process border radius (rounded corners)
            self.compute_border_radius(css, element)

            # Process stroke (border) properties
            self.compute_stroke(css, element)

            # Process effects.
            self.compute_effects(css, element)

        # Create CSS selector with parent hierarchy if applicable
        if parent and parent["name"] and not parent["name"].endswith(".html"):
            (_, _, _, parent_selector) = parse_name(parent["name"])
            selector = parent_selector + " " + selector

        # Initialize DOM element
        dom_element = None
        component = None

        if "componentId" in element:
            if element["componentId"] in self.components:
                component = self.components[element["componentId"]]
            else:
                print(f"Component {element['componentId']} not found")

        # Process based on element type
        if tag == "input" or tag == "textarea":
            dom_element = DomNode(tag)
            if len(clss) > 0:
                dom_element.attributes["name"] = clss[0]

            # Look for placeholder text which is what we will use as the text style.
            if "children" in element:
                for child in element["children"]:
                    if child["type"] == "TEXT":
                        placeholder_css = {}
                        self.compute_text_properties(placeholder_css, child)
                        dom_element.attributes["placeholder"] = child.get("characters", "").replace("\u2028", "\n")
                        css_rules.append({"selector": selector + "::placeholder", "styles": placeholder_css})
                        # Copy some of the rules:
                        self.compute_text_properties(css, child)
                        for fill in child["fills"]:
                            css["color"] = rgba(fill["color"])

            # Add CSS rule
            css_rules.append({"selector": selector, "styles": css})

        elif element_type == "IMAGE" or "exportSettings" in element or (component and "exportSettings" in component):
            # Create an img
            if component:
                src = self.data_dir + "/" + component["name"] + ".png"
            else:
                src = f"{self.data_dir}/{element_name.replace(' ', '_')}.png"
            dom_element = DomNode("img", {"src": src, "alt": element_name})

            # Add CSS rule
            css_rules.append({"selector": selector, "styles": css})

        elif element_type == "FRAME":
            # Create a div for the frame
            dom_element = DomNode(tag)

            # Add CSS rule
            css_rules.append({"selector": selector, "styles": css})

            # Process children
            if "children" in element:
                for child in element["children"]:
                    # Skip groups but process their children
                    if child.get("type") == "GROUP":
                        if "children" in child:
                            for group_child in child["children"]:
                                child_element, child_css = self.process_element(group_child, element)
                                dom_element.append_child(child_element)
                                css_rules.extend(child_css)
                    else:
                        child_element, child_css = self.process_element(child, element)
                        dom_element.append_child(child_element)
                        css_rules.extend(child_css)

        elif element_type == "TEXT":
            # Get text content
            text_content = element.get("characters", "").replace("\u2028", "<br>")

            # Create a span for the text
            dom_element = DomNode("span", {}, text_content)

            # Add CSS rule
            css["display"] = "inline-block"
            css_rules.append({"selector": selector, "styles": css})

        elif element_type == "RECTANGLE" or element_type == "VECTOR" or element_type == "ELLIPSE":
            # Check if it's an image
            if "fills" in element and element["fills"] and element["fills"][0].get("type") == "IMAGE":
                img_src = f"{self.data_dir}/{element_name.replace(' ', '_')}.png"
                dom_element = DomNode("img", {"src": img_src, "alt": element_name})

                # Add CSS rule
                css_rules.append({"selector": selector, "styles": css})
            else:
                # Create a div for shapes
                dom_element = DomNode(tag)

                # Add CSS rule
                css_rules.append({"selector": selector, "styles": css})

        else:
            # For other types, create a simple div
            dom_element = DomNode(tag)

            # Add CSS rule
            css_rules.append({"selector": selector, "styles": css})

            # Process children
            if "children" in element:
                for child in element["children"]:
                    child_element, child_css = self.process_element(child, element)
                    dom_element.append_child(child_element)
                    css_rules.extend(child_css)

        if id:
            dom_element.attributes["id"] = id
        if len(clss) > 0:
            dom_element.attributes["class"] = " ".join(clss)

        return dom_element, css_rules

    def traverse_nodes(self, node: Dict[str, Any], callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Recursively traverse nodes in a Figma document.

        Args:
            node: Current node to traverse
            callback: Function to call on each node
        """
        # Call the callback on this node
        callback(node)

        # Check if node has children
        if "children" in node:
            for child in node["children"]:
                self.traverse_nodes(child, callback)

    def process_document(self, document: Dict[str, Any], url: str) -> None:
        """
        Process the Figma document to find frames ending with .html.

        Args:
            document: Figma document data
            url: URL of the Figma file
        """

        # Get the document node
        doc_node = document.get("document", {})

        self.components = {}

        def collect_components(node):
            # Check if the node is a component then add it to the components
            if node["type"] == "COMPONENT":
                self.components[node["id"]] = node

        self.traverse_nodes(doc_node, collect_components)

        def process_node(node):
            # Check if node is a frame and name ends with .html
            if node.get("type") == "FRAME" and node.get("name", "").endswith(".html"):
                self.gen_html_page(node, url)

        self.traverse_nodes(doc_node, process_node)

    def gen_html(self, token: str, figma_url: str) -> dict:
        """
        Generate HTML from a Figma file.

        Args:
            token: Figma API token
            url: URL of the Figma file

        Returns:
            dict: Figma document data
        """

        headers = {"X-Figma-Token": token}

        file_id = figma_url.split("/")[-1].split("?")[0]

        api_url = f"https://api.figma.com/v1/files/{file_id}"

        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            document = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Figma file: {e}", file=sys.stderr)
            sys.exit(1)

        # Write data to file
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        with open(f"{self.tmp_dir}/figma_data.json", "w") as f:
            json.dump(document, f, indent=2)

        # Process the document to find frames and generate HTML
        self.process_document(document, figma_url)

        return document


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Figma file to HTML with optional extras.")

    parser.add_argument(
        "--input_url",
        help="URL of the input Figma file.",
        default="https://www.figma.com/design/WeQldl3PsqFIpDnTka5Kd3",
    )
    parser.add_argument("--output_dir", help="Directory to write output files to.", default=".")

    parser.add_argument("--tmp_dir", help="Temporary directory for intermediate files.", default="dist")
    parser.add_argument("--extra-css", dest="extra_css", help="Path to extra CSS file to include.", default="style.css")
    parser.add_argument("--extra-js", dest="extra_js", help="Path to extra JS file to include.", default="dist/main.js")
    parser.add_argument("--data-dir", dest="data_dir", help="Root directory for data files.", default="data")

    return parser.parse_args()


if __name__ == "__main__":
    token = open(os.path.expanduser("~/.figma_token")).read().strip()
    args = parse_args()

    generator = HtmlGenerator(
        output_dir=args.output_dir,
        extra_css=args.extra_css,
        extra_js=args.extra_js,
        data_dir=args.data_dir,
        tmp_dir=args.tmp_dir,
    )
    generator.gen_html(token, args.input_url)
