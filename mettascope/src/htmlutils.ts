
// Parse a hex color string into a float array.
export function parseHtmlColor(color: string): [number, number, number, number] {
  return [
    parseInt(color.slice(1, 3), 16) / 255,
    parseInt(color.slice(3, 5), 16) / 255,
    parseInt(color.slice(5, 7), 16) / 255,
    1.0
  ];
}

// Find an element by css selector.
export function find(selector: string): HTMLElement {
  const elements = document.querySelectorAll(selector);
  if (elements.length === 0) {
    throw new Error(`Element with selector "${selector}" not found`);
  } else if (elements.length > 1) {
    throw new Error(`Multiple elements with selector "${selector}" found`);
  }
  return elements[0] as HTMLElement;
}

// Find multiple elements by css selector.
export function finds(selector: string): HTMLElement[] {
  return Array.from(document.querySelectorAll(selector));
}
