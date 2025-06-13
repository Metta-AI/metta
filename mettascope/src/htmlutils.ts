/** Parse a hex color string into a float array. */
export function parseHtmlColor(color: string): [number, number, number, number] {
  return [
    parseInt(color.slice(1, 3), 16) / 255,
    parseInt(color.slice(3, 5), 16) / 255,
    parseInt(color.slice(5, 7), 16) / 255,
    1.0
  ];
}

/** Find an element by css selector in a parent element. */
export function findIn(parent: HTMLElement, selector: string): HTMLElement {
  const elements = parent.querySelectorAll(selector);
  if (elements.length === 0) {
    throw new Error(`Element with selector "${selector}" not found`);
  } else if (elements.length > 1) {
    throw new Error(`Multiple elements with selector "${selector}" found`);
  }
  return elements[0] as HTMLElement;
}

/** Find an element by css selector. */
export function find(selector: string): HTMLElement {
  return findIn(window.document.body, selector);
}

/** Find multiple elements by css selector. */
export function finds(selector: string): HTMLElement[] {
  return Array.from(document.querySelectorAll(selector));
}

/** Remove all children of an element. */
export function removeChildren(element: HTMLElement) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

/** Walks up the DOM tree and finds the given attribute. */
export function findAttr(element: HTMLElement, attribute: string): string {
  var e = element;
  while (e != null && e != document.body) {
    if (e.hasAttribute(attribute)) {
      return e.getAttribute(attribute) as string;
    }
    e = e.parentElement as HTMLElement;
  }
  return "";
}


/**
 * I don't like the way that DOM handles events. I think they should be done
 * via CSS selectors rather then handlers one attaches and detaches all the time.
 * Here we have an onEvent function that attaches to a CSS selector and
 * is called when that event happens.
 */

type Handler = {
  selector: string;
  event: string;
  callback: (target: HTMLElement, event: Event) => void;
}

var globalHandlers: Map<string, Handler[]> = new Map();

export function onEvent(event: string, selector: string, callback: (target: HTMLElement, event: Event) => void) {
  let handler: Handler = {
    selector: selector,
    event: event,
    callback: callback
  };
  if (!globalHandlers.has(event)) {
    // First time we've seen this event.
    window.addEventListener(event, (e: Event) => {
      if (event == "click") {
        hideMenu();
      }
      let handlers = globalHandlers.get(event);
      if (handlers) {
        var target = e.target as HTMLElement;
        while (target != null) {
          for (let handler of handlers) {
            if (target.matches(handler.selector)) {
              handler.callback(target, e);
              return;
            }
          }
          target = target.parentElement as HTMLElement;
        }
      }
    }, { passive: false })
    globalHandlers.set(event, []);
  }
  globalHandlers.get(event)?.push(handler);
}


/**
 * Menus are hidden on any click outside the menu, and are clicked through.
 * Dropdowns are hidden on any click outside the dropdown, and are not clicked
 * through, as they have a scrim.
 */

var openMenuTarget: HTMLElement | null = null;
var openMenu: HTMLElement | null = null;

/** Shows a menu and sets the scrim target to the menu. */
export function showMenu(target: HTMLElement, menu: HTMLElement) {

  // Hide any other open menu.
  hideMenu();

  // Get location of the target.
  openMenuTarget = target;
  openMenu = menu;
  openMenuTarget.classList.add("selected");
  let rect = openMenuTarget.getBoundingClientRect();
  openMenu.style.left = rect.left + "px";
  openMenu.style.top = (rect.bottom + 2) + "px";
  openMenu.classList.remove("hidden");
  // Bring menu to front (move to the end of the sibling list)
  openMenu.parentElement?.appendChild(openMenu);
}

/** Hides the menu and the scrim. */
export function hideMenu() {
  if (openMenuTarget != null) {
    openMenuTarget.classList.remove("selected");
    openMenuTarget = null;
  }
  if (openMenu != null) {
    openMenu.classList.add("hidden");
    openMenu = null;
  }
}

var openDropdownTarget: HTMLElement | null = null;
var openDropdown: HTMLElement | null = null;
var scrim = find('#scrim') as HTMLDivElement;
var scrimTarget: HTMLElement | null = null;
scrim.classList.add("hidden");

onEvent("click", "#scrim", (target: HTMLElement, event: Event) => {
  hideMenu();
  hideDropdown();
});

/** Shows a dropdown and sets the scrim target to the dropdown. */
export function showDropdown(target: HTMLElement, dropdown: HTMLElement) {
  hideDropdown();
  openDropdown = dropdown;
  openDropdownTarget = target;
  let rect = openDropdownTarget.getBoundingClientRect();
  openDropdown.style.left = rect.left + "px";
  openDropdown.style.top = (rect.bottom + 2) + "px";
  openDropdown.classList.remove("hidden");
  scrim.classList.remove("hidden");
}

/** Hides the dropdown and the scrim. */
export function hideDropdown() {
  if (openDropdownTarget != null) {
    openDropdownTarget.classList.remove("selected");
    openDropdownTarget = null;
  }
  if (openDropdown != null) {
    openDropdown.classList.add("hidden");
    openDropdown = null;
  }
  scrim.classList.add("hidden");
}


/** Get number out of local storage with a default value. */
export function localStorageGetNumber(key: string, defaultValue: number): number {
  let value = localStorage.getItem(key);
  if (value == null) {
    return defaultValue;
  }
  return parseFloat(value);
}

/** Set number in local storage. */
export function localStorageSetNumber(key: string, value: number) {
  localStorage.setItem(key, value.toString());
}

/** Get a whole data structure from local storage. */
export function localStorageGetObject<T>(key: string, defaultValue: T): T {
  let value = localStorage.getItem(key);
  if (value == null) {
    return defaultValue;
  }
  return JSON.parse(value);
}

/** Set a whole data structure in local storage. */
export function localStorageSetObject<T>(key: string, value: T) {
  localStorage.setItem(key, JSON.stringify(value));
}
