import { Vec2f, Mat3f } from './vector_math.js'

// Custom layout system for managing tabs and panes.

export class Tab {
  public title: string
  public content: string
  public isActive: boolean = false

  constructor(title: string, content: string) {
    this.title = title
    this.content = content
  }
}

export class Pane {
  public tabs: Tab[] = []
  public element: HTMLElement
  private tabBarElement!: HTMLElement
  private contentElement!: HTMLElement

  constructor(container: HTMLElement) {
    this.element = container
    this.render()
  }

  public addTab(tab: Tab): void {
    this.tabs.push(tab)
    if (this.tabs.length === 1) {
      tab.isActive = true
    }
    this.updateTabs()
  }

  private render(): void {
    this.element.innerHTML = `
            <div class="pane">
                <div class="tab-bar"></div>
                <div class="tab-content"></div>
            </div>
        `

    this.tabBarElement = this.element.querySelector('.tab-bar') as HTMLElement
    this.contentElement = this.element.querySelector('.tab-content') as HTMLElement

    this.addStyles()
  }

  private addStyles(): void {
    const style = document.createElement('style')
    style.textContent = `
            .pane {
                height: 100%;
                display: flex;
                flex-direction: column;
            }

            .tab-bar {
                background: #2d2d30;
                border-bottom: 1px solid #3c3c3c;
                display: flex;
                min-height: 32px;
            }

            .tab {
                padding: 8px 16px;
                background: #2d2d30;
                border-right: 1px solid #3c3c3c;
                cursor: pointer;
                user-select: none;
                color: #cccccc;
            }

            .tab.active {
                background: #1e1e1e;
                color: #ffffff;
            }

            .tab:hover:not(.active) {
                background: #37373d;
            }

            .tab-content {
                flex: 1;
                padding: 16px;
                background: #1e1e1e;
                overflow: auto;
            }
        `
    document.head.appendChild(style)
  }

  private updateTabs(): void {
    this.tabBarElement.innerHTML = ''

    this.tabs.forEach((tab, index) => {
      const tabElement = document.createElement('div')
      tabElement.className = `tab ${tab.isActive ? 'active' : ''}`
      tabElement.textContent = tab.title
      tabElement.addEventListener('click', () => this.activateTab(index))
      this.tabBarElement.appendChild(tabElement)
    })

    this.updateContent()
  }

  private activateTab(index: number): void {
    this.tabs.forEach((tab, i) => {
      tab.isActive = i === index
    })
    this.updateTabs()
  }

  private updateContent(): void {
    const activeTab = this.tabs.find((tab) => tab.isActive)
    this.contentElement.textContent = activeTab ? activeTab.content : ''
  }
}

// Initialize the layout system with a test tab.
export function initLayout(): void {
  const container = document.getElementById('layout-container')
  if (!container) {
    console.error('Layout container not found')
    return
  }

  const pane = new Pane(container)

  // Add a test tab to demonstrate the functionality.
  const testTab = new Tab('Welcome', 'This is the content of the first tab!')
  pane.addTab(testTab)
}

// Auto-initialize when the DOM is ready.
document.addEventListener('DOMContentLoaded', initLayout)
