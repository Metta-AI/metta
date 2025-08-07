import { find } from './htmlutils.js'

/** Interface for warning toast configuration. */
interface WarningToastConfig {
  message: string
  type?: 'warning' | 'error' | 'info'
  id?: string
}

/** Class to manage warning toasts that persist until manually dismissed. */
class WarningToastManager {
  private toastContainer!: HTMLElement
  private activeToasts: Map<string, HTMLElement> = new Map()
  private toastCounter = 0

  constructor() {
    this.createToastContainer()
  }

  /** Creates the container for warning toasts. */
  private createToastContainer() {
    this.toastContainer = document.createElement('div')
    this.toastContainer.id = 'warning-toast-container'
    this.toastContainer.className = 'warning-toast-container'
    document.body.appendChild(this.toastContainer)
  }

  /** Shows a warning toast that persists until dismissed. */
  showWarningToast(config: WarningToastConfig): string {
    const toastId = config.id || `warning-toast-${++this.toastCounter}`

    // Remove existing toast with same id if it exists
    if (this.activeToasts.has(toastId)) {
      this.dismissToast(toastId)
    }

    const toast = this.createToastElement(config, toastId)
    this.toastContainer.appendChild(toast)
    this.activeToasts.set(toastId, toast)

    // Trigger the animation
    requestAnimationFrame(() => {
      toast.classList.add('show')
    })

    return toastId
  }

  /** Creates the HTML element for a warning toast. */
  private createToastElement(config: WarningToastConfig, toastId: string): HTMLElement {
    const toast = document.createElement('div')
    toast.className = `warning-toast warning-toast-${config.type || 'warning'}`
    toast.setAttribute('data-toast-id', toastId)

    const icon = this.getIconForType(config.type || 'warning')

    toast.innerHTML = `
      <div class="warning-toast-content">
        <div class="warning-toast-icon">${icon}</div>
        <div class="warning-toast-message">${config.message}</div>
        <button class="warning-toast-close" type="button" aria-label="Close warning">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M12.854 3.146a.5.5 0 0 0-.708 0L8 7.293 3.854 3.146a.5.5 0 1 0-.708.708L7.293 8l-4.147 4.146a.5.5 0 0 0 .708.708L8 8.707l4.146 4.147a.5.5 0 0 0 .708-.708L8.707 8l4.147-4.146a.5.5 0 0 0 0-.708z"/>
          </svg>
        </button>
      </div>
    `

    // Add click handler for close button
    const closeButton = toast.querySelector('.warning-toast-close') as HTMLElement
    closeButton.addEventListener('click', () => {
      this.dismissToast(toastId)
    })

    return toast
  }

  /** Gets the appropriate icon for the toast type. */
  private getIconForType(type: string): string {
    switch (type) {
      case 'error':
        return `
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
          </svg>
        `
      case 'info':
        return `
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
          </svg>
        `
      case 'warning':
      default:
        return `
          <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
          </svg>
        `
    }
  }

  /** Dismisses a warning toast by its ID. */
  dismissToast(toastId: string): void {
    const toast = this.activeToasts.get(toastId)
    if (!toast) return

    toast.classList.add('hide')

    // Remove after animation completes
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast)
      }
      this.activeToasts.delete(toastId)
    }, 300)
  }

  /** Dismisses all active warning toasts. */
  dismissAllToasts(): void {
    for (const toastId of this.activeToasts.keys()) {
      this.dismissToast(toastId)
    }
  }

  /** Gets the number of active warning toasts. */
  getActiveToastCount(): number {
    return this.activeToasts.size
  }
}

// Create a singleton instance
const warningToastManager = new WarningToastManager()

/** Shows a warning toast that persists until manually dismissed. */
export function showWarningToast(message: string, type: 'warning' | 'error' | 'info' = 'warning', id?: string): string {
  return warningToastManager.showWarningToast({ message, type, id })
}

/** Dismisses a warning toast by its ID. */
export function dismissWarningToast(toastId: string): void {
  warningToastManager.dismissToast(toastId)
}

/** Dismisses all active warning toasts. */
export function dismissAllWarningToasts(): void {
  warningToastManager.dismissAllToasts()
}

/** Gets the number of active warning toasts. */
export function getActiveWarningToastCount(): number {
  return warningToastManager.getActiveToastCount()
}
