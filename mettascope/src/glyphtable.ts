/**
 * This file handles the glyph editor modal.
 * It allows users to edit associations between glyph IDs (0-255) and string descriptions.
 */

import { find, onEvent } from './htmlutils.js'

// Store glyph associations in memory
export interface GlyphAssociation {
  id: number
  description: string
}

// Default glyph associations (can be customized)
const defaultGlyphAssociations: string[] = Array(256).fill('')
defaultGlyphAssociations[1] = '↑' // up
defaultGlyphAssociations[2] = '↓' // down
defaultGlyphAssociations[3] = '→' // right
defaultGlyphAssociations[4] = '←' // left
// Add more as needed

let glyphAssociations: string[] = []

function loadGlyphAssociations() {
  const saved = localStorage.getItem('glyphAssociations')
  if (saved) {
    try {
      const parsed = JSON.parse(saved)
      if (Array.isArray(parsed)) {
        glyphAssociations = parsed
      } else {
        throw new Error('Parsed glyphAssociations is not an array')
      }
    } catch (e) {
      console.error('Failed to load glyph associations:', e)
      glyphAssociations = [...defaultGlyphAssociations]
    }
  } else {
    glyphAssociations = [...defaultGlyphAssociations]
  }
}

// Save glyph associations to localStorage
function saveGlyphAssociations() {
  const data = Array.from(glyphAssociations.entries())
  localStorage.setItem('glyphAssociations', JSON.stringify(data))
}

// Get glyph description
export function getGlyphDescription(id: number): string {
  return glyphAssociations[id] || ''
}

// Set glyph description
export function setGlyphDescription(id: number, description: string) {
  if (description.length > 20) {
    description = description.substring(0, 20)
  }

  glyphAssociations[id] = description || ''

  saveGlyphAssociations()
}

// Generate table rows for all glyphs
function generateGlyphTableRows(filter: string = '') {
  const tbody = find('#glyph-table-body')
  tbody.innerHTML = ''

  const lowerFilter = filter.toLowerCase()

  for (let i = 0; i < 256; i++) {
    const description = getGlyphDescription(i)

    // Apply filter
    if (filter && !i.toString().includes(filter) && !description.toLowerCase().includes(lowerFilter)) {
      continue
    }

    const row = document.createElement('tr')
    row.innerHTML = `
      <td>${i}</td>
      <td><div class="glyph-preview" data-glyph-id="${i}">${String.fromCharCode(i)}</div></td>
      <td>
        <input type="text"
               class="glyph-description-input"
               data-glyph-id="${i}"
               value="${description}"
               maxlength="20"
               placeholder="Enter description...">
      </td>
      <td>
        <button class="glyph-action-button save" data-glyph-id="${i}">Save</button>
        <button class="glyph-action-button reset" data-glyph-id="${i}">Reset</button>
      </td>
    `
    tbody.appendChild(row)
  }
}

// Show the glyph editor modal
export function showGlyphEditor() {
  const modal = find('#glyph-editor-modal')
  modal.classList.remove('hidden')
  generateGlyphTableRows()

  // Focus search input
  const searchInput = find('#glyph-search') as HTMLInputElement
  searchInput.value = ''
  searchInput.focus()
}

// Hide the glyph editor modal
export function hideGlyphEditor() {
  const modal = find('#glyph-editor-modal')
  modal.classList.add('hidden')
}

// Initialize the glyph editor
export function initGlyphTable() {
  loadGlyphAssociations()

  // If the button exists in HTML, just add the modal
  // Otherwise, inject everything dynamically
  if (document.getElementById('glyph-toggle')) {
    injectGlyphEditorModal()
  } else {
    injectGlyphEditorHTML()
  }

  // Toggle button handler - only if we injected it
  if (!document.getElementById('glyph-toggle')) {
    onEvent('click', '#glyph-editor-toggle', () => {
      showGlyphEditor()
    })
  }

  // Close button handler
  onEvent('click', '#glyph-editor-modal .close-button', () => {
    hideGlyphEditor()
  })

  // Click outside modal to close
  onEvent('click', '#glyph-editor-modal', (target: HTMLElement, e: Event) => {
    if (target.id === 'glyph-editor-modal') {
      hideGlyphEditor()
    }
  })

  // Search input handler
  onEvent('input', '#glyph-search', (target: HTMLElement) => {
    const input = target as HTMLInputElement
    generateGlyphTableRows(input.value)
  })

  // Save button handler
  onEvent('click', '.glyph-action-button.save', (target: HTMLElement) => {
    const glyphId = parseInt(target.getAttribute('data-glyph-id') || '0')
    const input = document.querySelector(`.glyph-description-input[data-glyph-id="${glyphId}"]`) as HTMLInputElement
    if (input) {
      setGlyphDescription(glyphId, input.value.trim())
      // Visual feedback
      target.textContent = 'Saved!'
      setTimeout(() => {
        target.textContent = 'Save'
      }, 1000)
    }
  })

  // Reset button handler
  onEvent('click', '.glyph-action-button.reset', (target: HTMLElement) => {
    const glyphId = parseInt(target.getAttribute('data-glyph-id') || '0')
    const defaultDesc = defaultGlyphAssociations[glyphId] || ''
    setGlyphDescription(glyphId, defaultDesc)

    const input = document.querySelector(`.glyph-description-input[data-glyph-id="${glyphId}"]`) as HTMLInputElement
    if (input) {
      input.value = defaultDesc
    }
  })

  // Reset all button handler
  onEvent('click', '#glyph-reset-all', () => {
    if (confirm('Are you sure you want to reset all glyph descriptions to defaults?')) {
      glyphAssociations = [...defaultGlyphAssociations]
      saveGlyphAssociations()
      generateGlyphTableRows()
    }
  })

  // Export handler
  onEvent('click', '#glyph-export', () => {
    const sparseData: Record<number, string> = {}
    glyphAssociations.forEach((desc, id) => {
      if (desc && desc.trim()) {
        sparseData[id] = desc
      }
    })

    const json = JSON.stringify(sparseData, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'glyph_associations.json'
    a.click()
    URL.revokeObjectURL(url)
  })

  // Import handler
  onEvent('click', '#glyph-import', () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (file) {
        try {
          const text = await file.text()
          const data = JSON.parse(text)
          if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
            const newAssociations = Array(256).fill('')
            for (const [key, value] of Object.entries(data)) {
              const id = parseInt(key, 10)
              if (!isNaN(id) && typeof value === 'string') {
                newAssociations[id] = value
              }
            }
            glyphAssociations = newAssociations
            saveGlyphAssociations()
            generateGlyphTableRows()
          } else {
            throw new Error('Expected a JSON object')
          }
        } catch (e) {
          alert('Failed to import file. Please ensure it is a valid JSON object.')
        }
      }
    }
    input.click()
  })

  // Handle Enter key to save
  onEvent('keydown', '.glyph-description-input', (target: HTMLElement, e: Event) => {
    const keyEvent = e as KeyboardEvent
    if (keyEvent.key === 'Enter') {
      const input = target as HTMLInputElement
      const glyphId = parseInt(input.getAttribute('data-glyph-id') || '0')
      setGlyphDescription(glyphId, input.value.trim())

      // Find and trigger the save button for visual feedback
      const saveButton = document.querySelector(`.glyph-action-button.save[data-glyph-id="${glyphId}"]`) as HTMLElement
      if (saveButton) {
        saveButton.click()
      }
    }
  })

  // Handle Escape key to close modal
  onEvent('keydown', 'body', (target: HTMLElement, e: Event) => {
    const keyEvent = e as KeyboardEvent
    if (keyEvent.key === 'Escape') {
      const modal = find('#glyph-editor-modal')
      if (!modal.classList.contains('hidden')) {
        hideGlyphEditor()
        e.preventDefault()
        e.stopPropagation()
      }
    }
  })
}

// Inject just the modal if button exists in HTML
function injectGlyphEditorModal() {
  // Add the modal to the body
  const modalHTML = `
    <div id="glyph-editor-modal" class="modal hidden">
      <div class="modal-content glyph-editor-content">
        <div class="modal-header">
          <h2>Glyph Editor</h2>
          <button class="close-button">&times;</button>
        </div>
        <div class="modal-body">
          <div class="glyph-editor-controls">
            <input type="text" id="glyph-search" placeholder="Search glyphs..." class="glyph-search">
            <button id="glyph-reset-all" class="glyph-button">Reset All</button>
            <button id="glyph-export" class="glyph-button">Export</button>
            <button id="glyph-import" class="glyph-button">Import</button>
          </div>
          <div class="glyph-table-container">
            <table class="glyph-table">
              <thead>
                <tr>
                  <th>Glyph ID</th>
                  <th>Preview</th>
                  <th>Description</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="glyph-table-body">
                <!-- Rows will be dynamically generated -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  `
  document.body.insertAdjacentHTML('beforeend', modalHTML)

  // Inject the CSS styles
  const styleElement = document.createElement('style')
  styleElement.textContent = GLYPH_EDITOR_STYLES
  document.head.appendChild(styleElement)
}

// Dynamically inject the glyph editor HTML to avoid Figma overwrites
function injectGlyphEditorHTML() {
  // Add the toggle button to the header
  const helpButton = find('#help-button')
  if (helpButton) {
    const glyphButton = document.createElement('button')
    glyphButton.id = 'glyph-editor-toggle'
    glyphButton.className = 'hover-icon'
    glyphButton.title = 'Edit glyph associations'
    glyphButton.innerHTML = '<img src="data/ui/glyphs.png" alt="icon" class="icon">'
    helpButton.parentElement?.insertBefore(glyphButton, helpButton.nextSibling)
  }

  // Add the modal
  injectGlyphEditorModal()
}

// CSS styles as a string constant
const GLYPH_EDITOR_STYLES = `
/* Glyph Editor Modal Styles */
.glyph-editor-content {
  width: 90%;
  max-width: 800px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background-color: #2a2a2a;
  border-bottom: 1px solid #444;
}

.modal-header h2 {
  margin: 0;
  color: #fff;
  font-size: 1.5em;
}

.close-button {
  background: none;
  border: none;
  color: #fff;
  font-size: 24px;
  cursor: pointer;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.close-button:hover {
  background-color: rgba(255, 255, 255, 0.1);
}

.modal-body {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.glyph-editor-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  align-items: center;
}

.glyph-search {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #444;
  border-radius: 4px;
  background-color: #1a1a1a;
  color: #fff;
  font-size: 14px;
}

.glyph-search:focus {
  outline: none;
  border-color: #666;
}

.glyph-button {
  padding: 8px 16px;
  background-color: #3a3a3a;
  color: #fff;
  border: 1px solid #555;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
}

.glyph-button:hover {
  background-color: #4a4a4a;
}

.glyph-table-container {
  flex: 1;
  overflow-y: auto;
  border: 1px solid #444;
  border-radius: 4px;
  background-color: #1a1a1a;
}

.glyph-table {
  width: 100%;
  border-collapse: collapse;
}

.glyph-table th {
  position: sticky;
  top: 0;
  background-color: #2a2a2a;
  color: #fff;
  padding: 12px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #444;
  z-index: 10;
}

.glyph-table td {
  padding: 8px 12px;
  border-bottom: 1px solid #333;
  color: #ddd;
}

.glyph-table tbody tr:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.glyph-table th:first-child,
.glyph-table td:first-child {
  width: 80px;
  text-align: center;
}

.glyph-table th:nth-child(2),
.glyph-table td:nth-child(2) {
  width: 80px;
  text-align: center;
}

.glyph-table th:last-child,
.glyph-table td:last-child {
  width: 100px;
  text-align: center;
}

.glyph-preview {
  display: inline-block;
  width: 32px;
  height: 32px;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 4px;
  text-align: center;
  line-height: 32px;
  font-family: monospace;
  font-size: 16px;
  color: #fff;
}

.glyph-description-input {
  width: 100%;
  max-width: 300px;
  padding: 6px 10px;
  border: 1px solid #444;
  border-radius: 4px;
  background-color: #2a2a2a;
  color: #fff;
  font-size: 14px;
}

.glyph-description-input:focus {
  outline: none;
  border-color: #666;
  background-color: #333;
}

.glyph-action-button {
  padding: 4px 12px;
  margin: 0 2px;
  background-color: #3a3a3a;
  color: #fff;
  border: 1px solid #555;
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
  transition: background-color 0.2s;
}

.glyph-action-button:hover {
  background-color: #4a4a4a;
}

.glyph-action-button.save {
  background-color: #2a5a2a;
  border-color: #3a6a3a;
}

.glyph-action-button.save:hover {
  background-color: #3a6a3a;
}

.glyph-action-button.reset {
  background-color: #5a2a2a;
  border-color: #6a3a3a;
}

.glyph-action-button.reset:hover {
  background-color: #6a3a3a;
}
`

// Export the associations for use in other parts of the application
export function getGlyphAssociations(): Array<string> {
  return [...glyphAssociations]
}
