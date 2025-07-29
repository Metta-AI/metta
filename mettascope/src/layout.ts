import { Vec2f, Mat3f } from './vector_math.js'

// Custom layout system for managing tabs and panes.

export enum PanelType {
  LOGS = 'Logs',
  METRICS = 'Metrics',
  MAP_VIEW = 'Map View',
  AGENT_DETAILS = 'Agent Details'
}

export class Tab {
  public title: string
  public content: string
  public isActive: boolean = false
  public panelType: PanelType

  constructor(title: string, content: string, panelType: PanelType = PanelType.LOGS) {
    this.title = title
    this.content = content
    this.panelType = panelType
  }
}

export class Pane {
  public tabs: Tab[] = []
  public element: HTMLElement
  private tabBarElement!: HTMLElement
  private contentElement!: HTMLElement
  private addTabContainer!: HTMLElement
  private dropdown!: HTMLElement
  private isDropdownVisible: boolean = false

  constructor(container: HTMLElement) {
    this.element = container
    this.render()
    this.setupEventListeners()
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
                <div class="tab-bar">
                    <div class="add-tab-container">
                        <div class="add-tab-btn">+</div>
                        <div class="panel-type-dropdown">
                            <div class="dropdown-item" data-type="${PanelType.LOGS}">${PanelType.LOGS}</div>
                            <div class="dropdown-item" data-type="${PanelType.METRICS}">${PanelType.METRICS}</div>
                            <div class="dropdown-item" data-type="${PanelType.MAP_VIEW}">${PanelType.MAP_VIEW}</div>
                            <div class="dropdown-item" data-type="${PanelType.AGENT_DETAILS}">${PanelType.AGENT_DETAILS}</div>
                        </div>
                    </div>
                </div>
                <div class="tab-content"></div>
            </div>
        `

    this.tabBarElement = this.element.querySelector('.tab-bar') as HTMLElement
    this.contentElement = this.element.querySelector('.tab-content') as HTMLElement
    this.addTabContainer = this.element.querySelector('.add-tab-container') as HTMLElement
    this.dropdown = this.element.querySelector('.panel-type-dropdown') as HTMLElement
  }

  private setupEventListeners(): void {
    const addBtn = this.addTabContainer.querySelector('.add-tab-btn') as HTMLElement
    addBtn.addEventListener('click', (e) => {
      e.stopPropagation()
      this.toggleDropdown()
    })

    // Handle dropdown item clicks.
    const dropdownItems = this.addTabContainer.querySelectorAll('.dropdown-item')
    dropdownItems.forEach(item => {
      item.addEventListener('click', (e) => {
        const panelType = (e.target as HTMLElement).getAttribute('data-type') as PanelType
        this.createNewTab(panelType)
        this.hideDropdown()
      })
    })

    // Close dropdown when clicking outside.
    document.addEventListener('click', (e) => {
      if (!this.addTabContainer.contains(e.target as Node)) {
        this.hideDropdown()
      }
    })
  }

  private toggleDropdown(): void {
    this.isDropdownVisible = !this.isDropdownVisible
    this.dropdown.classList.toggle('visible', this.isDropdownVisible)
  }

  private hideDropdown(): void {
    this.isDropdownVisible = false
    this.dropdown.classList.remove('visible')
  }

  private createNewTab(panelType: PanelType): void {
    const tabNumber = this.tabs.length + 1
    const content = this.generateContentForPanelType(panelType)
    const newTab = new Tab(`${panelType} ${tabNumber}`, content, panelType)
    this.addTab(newTab)
    this.activateTab(this.tabs.length - 1)
  }

  private generateContentForPanelType(panelType: PanelType): string {
    switch (panelType) {
      case PanelType.LOGS:
        return 'Log entries will appear here...\n\n[INFO] System started\n[DEBUG] Loading configuration\n[INFO] Ready to receive data'
      case PanelType.METRICS:
        return 'Performance metrics and statistics will be displayed here...\n\nCPU Usage: 45%\nMemory: 2.1GB / 8GB\nNetwork I/O: 1.2MB/s'
      case PanelType.MAP_VIEW:
        return 'Interactive map visualization will render here...\n\nMap dimensions: 100x100\nAgents: 5 active\nObstacles: 23'
      case PanelType.AGENT_DETAILS:
        return 'Agent status and details will be shown here...\n\nAgent ID: 001\nStatus: Active\nPosition: (45, 23)\nHealth: 100%'
      default:
        return 'Panel content will appear here...'
    }
  }

    private updateTabs(): void {
    // Clear existing tabs but keep the add-tab-container.
    const existingTabs = this.tabBarElement.querySelectorAll('.tab')
    existingTabs.forEach(tab => tab.remove())

    this.tabs.forEach((tab, index) => {
      const tabElement = document.createElement('div')
      tabElement.className = `tab ${tab.isActive ? 'active' : ''}`
      tabElement.textContent = tab.title
      tabElement.addEventListener('click', () => this.activateTab(index))

      // Insert before the add-tab-container.
      this.tabBarElement.insertBefore(tabElement, this.addTabContainer)
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
  const testTab = new Tab('Welcome', 'This is the content of the first tab!\n\nClick the + button to add more tabs.', PanelType.LOGS)
  pane.addTab(testTab)
}

// Auto-initialize when the DOM is ready.
document.addEventListener('DOMContentLoaded', initLayout)
