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
                <div class="tab-bar"></div>
                <div class="tab-content"></div>
            </div>
        `

    this.tabBarElement = this.element.querySelector('.tab-bar') as HTMLElement
    this.contentElement = this.element.querySelector('.tab-content') as HTMLElement

    // Create the add-tab-container separately.
    this.addTabContainer = document.createElement('div')
    this.addTabContainer.className = 'add-tab-container'
    this.addTabContainer.innerHTML = `
        <div class="add-tab-btn">+</div>
        <div class="panel-type-dropdown">
            <div class="dropdown-item" data-type="${PanelType.LOGS}">${PanelType.LOGS}</div>
            <div class="dropdown-item" data-type="${PanelType.METRICS}">${PanelType.METRICS}</div>
            <div class="dropdown-item" data-type="${PanelType.MAP_VIEW}">${PanelType.MAP_VIEW}</div>
            <div class="dropdown-item" data-type="${PanelType.AGENT_DETAILS}">${PanelType.AGENT_DETAILS}</div>
        </div>
    `

    this.dropdown = this.addTabContainer.querySelector('.panel-type-dropdown') as HTMLElement
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
    // Clear the entire tab bar.
    this.tabBarElement.innerHTML = ''

    // Add all tabs.
    this.tabs.forEach((tab, index) => {
      const tabElement = document.createElement('div')
      tabElement.className = `tab ${tab.isActive ? 'active' : ''}`
      tabElement.textContent = tab.title
      tabElement.addEventListener('click', () => this.activateTab(index))
      this.tabBarElement.appendChild(tabElement)
    })

    // Add the plus button after all tabs.
    this.tabBarElement.appendChild(this.addTabContainer)

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

export class SplitLayout {
  private container: HTMLElement
  private leftPane!: Pane
  private rightPane!: Pane
  private splitter!: HTMLElement
  private leftContainer!: HTMLElement
  private rightContainer!: HTMLElement
  private isDragging: boolean = false
  private startX: number = 0
  private startLeftWidth: number = 0

  constructor(container: HTMLElement) {
    this.container = container
    this.render()
    this.setupSplitter()
    this.initializePanes()
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="split-container">
        <div class="pane-container" style="flex: 1;"></div>
        <div class="splitter"></div>
        <div class="pane-container" style="flex: 1;"></div>
      </div>
    `

    this.leftContainer = this.container.querySelector('.pane-container:first-child') as HTMLElement
    this.rightContainer = this.container.querySelector('.pane-container:last-child') as HTMLElement
    this.splitter = this.container.querySelector('.splitter') as HTMLElement
  }

  private setupSplitter(): void {
    this.splitter.addEventListener('mousedown', (e) => {
      this.isDragging = true
      this.startX = e.clientX
      this.startLeftWidth = this.leftContainer.offsetWidth
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
      e.preventDefault()
    })

    document.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return

      const deltaX = e.clientX - this.startX
      const containerWidth = this.container.offsetWidth
      const splitterWidth = this.splitter.offsetWidth
      const newLeftWidth = this.startLeftWidth + deltaX
      const minWidth = 200
      const maxWidth = containerWidth - splitterWidth - minWidth

      if (newLeftWidth >= minWidth && newLeftWidth <= maxWidth) {
        const leftFlex = newLeftWidth / (containerWidth - splitterWidth)
        const rightFlex = 1 - leftFlex

        this.leftContainer.style.flex = `${leftFlex}`
        this.rightContainer.style.flex = `${rightFlex}`
      }
    })

    document.addEventListener('mouseup', () => {
      if (this.isDragging) {
        this.isDragging = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }
    })
  }

  private initializePanes(): void {
    // Create left pane with initial tab.
    this.leftPane = new Pane(this.leftContainer)
    const leftTab = new Tab('Left Panel', 'This is the left side panel.\n\nYou can add more tabs using the + button.', PanelType.LOGS)
    this.leftPane.addTab(leftTab)

    // Create right pane with initial tab.
    this.rightPane = new Pane(this.rightContainer)
    const rightTab = new Tab('Right Panel', 'This is the right side panel.\n\nIt works independently from the left panel.', PanelType.MAP_VIEW)
    this.rightPane.addTab(rightTab)
  }

  public getLeftPane(): Pane {
    return this.leftPane
  }

  public getRightPane(): Pane {
    return this.rightPane
  }
}

// Initialize the layout system with split panes.
export function initLayout(): void {
  const container = document.getElementById('layout-container')
  if (!container) {
    console.error('Layout container not found')
    return
  }

  const splitLayout = new SplitLayout(container)
}

// Auto-initialize when the DOM is ready.
document.addEventListener('DOMContentLoaded', initLayout)
