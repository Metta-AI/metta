// Tooltip titles for UI elements.

import { find } from './htmlutils.js'

interface TooltipConfig {
  [elementSelector: string]: string
}

const tooltipTitles: TooltipConfig = {
  '#rewind-to-start': 'Rewind to start',
  '#step-back': 'Step back',
  '#play': 'Play / Pause',
  '#step-forward': 'Step forward',
  '#rewind-to-end': 'Fast-forward to end',
  '#demo-mode-toggle': 'Toggle demo mode',
  '#full-screen-toggle': 'Toggle full-screen',
  '#speed1': 'Speed ×0.25',
  '#speed2': 'Speed ×0.5',
  '#speed3': 'Speed ×1',
  '#speed4': 'Speed ×2',
  '#speed5': 'Speed ×4',
  '#speed6': 'Speed ×8',
  '#focus-toggle': 'Toggle focus mode',
  '#minimap-toggle': 'Toggle minimap',
  '#controls-toggle': 'Show controls',
  '#info-toggle': 'Show info',
  '#agent-panel-toggle': 'Toggle agent panel',
  '#traces-toggle': 'Toggle traces',
  '#resources-toggle': 'Toggle resources',
  '#heatmap-toggle': 'Toggle heatmap',
  '#grid-toggle': 'Toggle grid',
  '#visual-range-toggle': 'Toggle visual range',
  '#fog-of-war-toggle': 'Toggle fog of war',
  '#share-button': 'Share replay',
  '#help-button': 'Help',
}

export function initializeTooltips(): void {
  for (const elementSelector in tooltipTitles) {
    const element = find(elementSelector)
    if (element) {
      element.title = tooltipTitles[elementSelector]
    }
  }
}
