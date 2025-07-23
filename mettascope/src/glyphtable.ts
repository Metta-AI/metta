/**
 * This file handles the glyph editor modal.
 * It allows users to edit associations between glyph IDs (0-255) and glyph strings.
 */

import { state } from './common.js'
import { find, onEvent } from './htmlutils.js'

// Default glyph associations (can be customized)
const defaultGlyphAssociations: string[] = Array(256).fill('')
defaultGlyphAssociations[1] = '↑' // up
defaultGlyphAssociations[2] = '↓' // down
defaultGlyphAssociations[3] = '→' // right
defaultGlyphAssociations[4] = '←' // left

let glyphAssociations: string[] = []

function loadGlyphAssociations() {
  const saved = localStorage.getItem('glyphAssociations')
  if (saved) {
    try {
      const parsed = JSON.parse(saved)
      if (Array.isArray(parsed)) {
        glyphAssociations = parsed
        // Ensure array has 256 elements
        while (glyphAssociations.length < 256) {
          glyphAssociations.push('')
        }
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
  localStorage.setItem('glyphAssociations', JSON.stringify(glyphAssociations))
}

// Track which glyphs have been modified
function isGlyphModified(id: number): boolean {
  return glyphAssociations[id] !== defaultGlyphAssociations[id]
}

// Get number of modified glyphs
function getModifiedCount(): number {
  let count = 0
  for (let i = 0; i < 256; i++) {
    if (isGlyphModified(i)) count++
  }
  return count
}

// Generate table rows for all glyphs
function generateGlyphTableRows(filter: string = '') {
  const tbody = find('#glyph-table-body')
  tbody.innerHTML = ''

  const lowerFilter = filter.toLowerCase()
  let visibleCount = 0

  for (let i = 0; i < 256; i++) {
    const glyphString = glyphAssociations[i] || ''

    if (filter) {
      const matchesId = i.toString().includes(lowerFilter)
      const matchesGlyphString = glyphString.toLowerCase().includes(lowerFilter)

      if (!matchesId && !matchesGlyphString) {
        continue
      }
    }

    visibleCount++
    const isModified = isGlyphModified(i)
    const row = document.createElement('tr')
    if (isModified) {
      row.classList.add('modified')
    }

    row.innerHTML = `
      <td class="glyph-id">${i}</td>

      <td class="glyph-description-cell">
        <input
          type="text"
          class="glyph-description-input"
          value="${glyphString}"
          data-glyph-id="${i}"
          placeholder="∅"
          maxlength="20"
        >
      </td>

      <td class="glyph-actions-cell">
        <button class="glyph-action-button save" data-glyph-id="${i}" title="Save (Enter)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"></path>
            <polyline points="17 21 17 13 7 13 7 21"></polyline>
            <polyline points="7 3 7 8 15 8"></polyline>
          </svg>
        </button>
        <button class="glyph-action-button reset" data-glyph-id="${i}" title="Reset to default">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="1 4 1 10 7 10"></polyline>
            <path d="M3.51 15a9 9 0 102.13-9.36L1 10"></path>
          </svg>
        </button>
      </td>
    `
    tbody.appendChild(row)
  }

  // Update counter
  updateGlyphCounter(visibleCount, filter)
}

// Update the glyph counter display
function updateGlyphCounter(visibleCount: number, filter: string) {
  const counter = find('#glyph-count')
  const modifiedCount = getModifiedCount()

  if (counter) {
    let text = ''
    if (filter) {
      text = `Showing ${visibleCount} of 256 glyphs`
    } else {
      text = 'All 256 glyphs'
    }

    if (modifiedCount > 0) {
      text += ` • ${modifiedCount} modified`
    }

    counter.textContent = text
  }
}

// Show the glyph editor modal
export function showGlyphEditorPanel() {
  const modal = find('#glyph-editor-modal')
  modal.classList.remove('hidden')
  generateGlyphTableRows()

  // Focus search input
  const searchInput = find('#glyph-search') as HTMLInputElement
  searchInput.value = ''
  searchInput.focus()
}

// Hide the glyph editor modal
export function hideGlyphEditorPanel() {
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
    console.error("can't find #glyph-toggle!")
  }

  // attach button handler - only if we injected it
  if (!document.getElementById('glyph-toggle')) {
    onEvent('click', '#glyph-editor-toggle', () => {
      showGlyphEditorPanel()
    })
  }

  // Search input handler with debouncing
  let searchTimeout: NodeJS.Timeout
  onEvent('input', '#glyph-search', (target: HTMLElement) => {
    const input = target as HTMLInputElement
    clearTimeout(searchTimeout)
    searchTimeout = setTimeout(() => {
      generateGlyphTableRows(input.value)
    }, 150)
  })

  // Save button handler
  onEvent('click', '.glyph-action-button.save', (target: HTMLElement) => {
    const glyphId = parseInt(target.getAttribute('data-glyph-id') || '0')
    const input = document.querySelector(`.glyph-description-input[data-glyph-id="${glyphId}"]`) as HTMLInputElement
    if (input) {
      glyphAssociations[glyphId] = input.value.trim()

      // Visual feedback
      target.classList.add('success')
      setTimeout(() => {
        target.classList.remove('success')
        // Update row style
        const row = target.closest('tr')
        if (row) {
          if (isGlyphModified(glyphId)) {
            row.classList.add('modified')
          } else {
            row.classList.remove('modified')
          }
        }
        // Update counter
        updateGlyphCounter(find('#glyph-table-body').querySelectorAll('tr').length, '')
      }, 500)
    }
  })

  // Reset button handler
  onEvent('click', '.glyph-action-button.reset', (target: HTMLElement) => {
    const glyphId = parseInt(target.getAttribute('data-glyph-id') || '0')
    const defaultDesc = defaultGlyphAssociations[glyphId] || ''
    glyphAssociations[glyphId] = defaultDesc

    const input = document.querySelector(`.glyph-description-input[data-glyph-id="${glyphId}"]`) as HTMLInputElement
    if (input) {
      input.value = defaultDesc
    }

    // Update row style
    const row = target.closest('tr')
    if (row) {
      row.classList.remove('modified')
    }

    // Update counter
    updateGlyphCounter(find('#glyph-table-body').querySelectorAll('tr').length, '')
  })

  // Reset all button handler
  onEvent('click', '#glyph-reset-all', () => {
    const modifiedCount = getModifiedCount()
    if (modifiedCount === 0) {
      alert('No modifications to reset.')
      return
    }

    if (
      confirm(`Are you sure you want to reset all ${modifiedCount} modified glyphs to defaults? This cannot be undone.`)
    ) {
      glyphAssociations = [...defaultGlyphAssociations]
      saveGlyphAssociations()
      generateGlyphTableRows()
    }
  })

  // Export handler
  onEvent('click', '#glyph-export', () => {
    const sparseData: Record<number, string> = {}
    let exportCount = 0

    glyphAssociations.forEach((desc, id) => {
      if (desc && desc.trim()) {
        sparseData[id] = desc
        exportCount++
      }
    })

    if (exportCount === 0) {
      alert('No glyph descriptions to export.')
      return
    }

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
            let importCount = 0

            for (const [key, value] of Object.entries(data)) {
              const id = parseInt(key, 10)
              if (!isNaN(id) && id >= 0 && id < 256 && typeof value === 'string') {
                newAssociations[id] = value.substring(0, 20)
                importCount++
              }
            }

            if (importCount === 0) {
              alert('No valid glyph associations found in file.')
              return
            }

            glyphAssociations = newAssociations
            saveGlyphAssociations()
            generateGlyphTableRows()

            alert(`Successfully imported ${importCount} glyph associations.`)
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
      glyphAssociations[glyphId] = input.value.trim()

      // Find and trigger the save button for visual feedback
      const saveButton = document.querySelector(`.glyph-action-button.save[data-glyph-id="${glyphId}"]`) as HTMLElement
      if (saveButton) {
        saveButton.click()
      }

      // Move to next input
      const nextId = glyphId + 1
      if (nextId < 256) {
        const nextInput = document.querySelector(
          `.glyph-description-input[data-glyph-id="${nextId}"]`
        ) as HTMLInputElement
        if (nextInput && nextInput.closest('tr')) {
          nextInput.focus()
          nextInput.select()
        }
      }
    }
  })

  // Handle Escape key to close modal
  onEvent('keydown', 'body', (target: HTMLElement, e: Event) => {
    const keyEvent = e as KeyboardEvent
    if (keyEvent.key === 'Escape') {
      const modal = find('#glyph-editor-modal')
      if (!modal.classList.contains('hidden')) {
        hideGlyphEditorPanel()
        e.preventDefault()
        e.stopPropagation()
      }
    }
  })

  // handle possible save show state
  if (state.showGlyphEditor) {
    showGlyphEditorPanel()
  }
}

// Inject just the modal if button exists in HTML
function injectGlyphEditorModal() {
  // Check if modal already exists
  if (document.getElementById('glyph-editor-modal')) {
    return
  }

  // Add the modal to the body
  const modalHTML = `
    <div id="glyph-editor-modal" class="modal hidden">
      <div class="modal-content glyph-editor-content">
        <div class="modal-header">
          <h2>Glyph Editor</h2>
          <span id="glyph-count" class="glyph-count">All 256 glyphs</span>
          <button class="close-button" title="Close (Esc)">&times;</button>
        </div>
        <div class="modal-body">
          <div class="glyph-editor-controls">
            <input type="text" id="glyph-search" placeholder="Search" class="glyph-search">
            <button id="glyph-reset-all" class="glyph-button" title="Reset all modified glyphs">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="1 4 1 10 7 10"></polyline>
                <path d="M3.51 15a9 9 0 102.13-9.36L1 10"></path>
              </svg>
              Reset All
            </button>
            <button id="glyph-export" class="glyph-button" title="Export">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Export
            </button>
            <button id="glyph-import" class="glyph-button" title="Import">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              Import
            </button>
          </div>
          <div class="glyph-table-container" tabindex="0">
            <table class="glyph-table">
              <colgroup>
                <col style="width: 50px;"> <!-- Fixed width for ID -->
                <col style="width: auto;"> <!-- Flexible width for input -->
                <col style="width: 90px;"> <!-- Shrink actions -->
              </colgroup>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Glyph</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="glyph-table-body">
                <!-- Rows will be dynamically generated -->
              </tbody>
            </table>
          </div>
          <div class="glyph-editor-footer">
            <span class="hint">Press Enter to save and move to next • Tab to navigate • Esc to close</span>
          </div>
        </div>
      </div>
    </div>
  `

  // Add the modal HTML
  document.body.insertAdjacentHTML('beforeend', modalHTML)

  // Add styles
  const styleElement = document.createElement('style')
  styleElement.textContent = GLYPH_EDITOR_STYLES
  document.head.appendChild(styleElement)

  const closeButton = document.querySelector('#glyph-editor-modal .close-button')
  if (closeButton) {
    closeButton.addEventListener('click', () => {
      ;(document.querySelector('#glyph-toggle') as HTMLButtonElement)?.click()
    })
  }
}

// CSS styles as a string constant
const GLYPH_EDITOR_STYLES = `
/* Glyph Editor Modal Styles */
.glyph-editor-content {
  width: 90%;
  max-width: 900px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  background-color: #1a1a1a;
  border: 1px solid #444;
  border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  background-color: #252525;
  border-bottom: 1px solid #444;
  border-radius: 8px 8px 0 0;
}

.modal-header h2 {
  margin: 0;
  color: #fff;
  font-size: 1.5em;
  font-weight: 600;
}

.glyph-count {
  color: #888;
  font-size: 0.9em;
  margin-left: auto;
  margin-right: 20px;
}

.close-button {
  background: none;
  border: none;
  color: #888;
  font-size: 28px;
  cursor: pointer;
  padding: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s;
}

.close-button:hover {
  background-color: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.modal-body {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  padding: 20px;
  padding-bottom: 10px;
}

.glyph-editor-controls {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  align-items: center;
  flex-wrap: wrap;
}

.glyph-search {
  flex: 1;
  min-width: 250px;
  padding: 10px 16px;
  border: 1px solid #3a3a3a;
  border-radius: 6px;
  background-color: #252525;
  color: #fff;
  font-size: 14px;
  transition: all 0.2s;
}

.glyph-search::placeholder {
  color: #666;
}

.glyph-search:focus {
  outline: none;
  border-color: #555;
  background-color: #2a2a2a;
}

.glyph-button {
  padding: 10px 18px;
  background-color: #2a2a2a;
  color: #ccc;
  border: 1px solid #3a3a3a;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.glyph-button:hover {
  background-color: #333;
  border-color: #444;
  color: #fff;
  transform: translateY(-1px);
}

.glyph-button:active {
  transform: translateY(0);
}

.glyph-button svg {
  width: 16px;
  height: 16px;
  stroke: currentColor;
}

.glyph-table-container {
  flex: 1;
  overflow-y: auto;
  overscroll-behavior: contain;
  -webkit-overflow-scrolling: touch;
  border: 1px solid #333;
  border-radius: 6px;
  background-color: #1a1a1a;
}

.glyph-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}

.glyph-table th {
  position: sticky;
  top: 0;
  background-color: #252525;
  color: #ccc;
  padding: 12px;
  text-align: left;
  font-weight: 600;
  border-bottom: 1px solid #444;
  z-index: 10;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.glyph-table td {
  padding: 8px 12px;
  border-bottom: 1px solid #2a2a2a;
  color: #ddd;
}

.glyph-table tbody tr {
  transition: background-color 0.1s;
}

.glyph-table tbody tr:hover {
  background-color: rgba(255, 255, 255, 0.02);
}

.glyph-table tbody tr.modified {
  background-color: rgba(100, 200, 100, 0.03);
}

.glyph-table tbody tr.modified:hover {
  background-color: rgba(100, 200, 100, 0.05);
}

.glyph-id {
  text-align: center;
  font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
  color: #666;
  font-size: 12px;
}

.glyph-preview-cell {
  text-align: center;
}

.glyph-preview {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background-color: #252525;
  border: 1px solid #333;
  border-radius: 6px;
  font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
  font-size: 18px;
  color: #fff;
  transition: all 0.15s;
  cursor: default;
}

.glyph-preview:hover {
  background-color: #2a2a2a;
  border-color: #444;
  transform: scale(1.05);
}

.glyph-preview.non-printable {
  color: #555;
  font-size: 14px;
}

.glyph-preview.null,
.glyph-preview.tab,
.glyph-preview.newline,
.glyph-preview.return,
.glyph-preview.space {
  color: #888;
  font-size: 20px;
}

.glyph-description-cell {
  padding-right: 20px;
}

.glyph-description-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #333;
  border-radius: 4px;
  background-color: #252525;
  color: #fff;
  font-size: 14px;
  transition: all 0.15s;
}

.glyph-description-input::placeholder {
  color: #555;
}

.glyph-description-input:focus {
  outline: none;
  border-color: #444;
  background-color: #2a2a2a;
}

.glyph-actions-cell {
  display: flex;
  gap: 6px;
  justify-content: center;
  align-items: center;
  white-space: nowrap;
  padding: 4px;
}

.glyph-action-button {
  padding: 6px 10px;
  margin: 0;
  background-color: #2a2a2a;
  color: #888;
  border: 1px solid #333;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.glyph-action-button:hover {
  background-color: #333;
  border-color: #444;
  color: #fff;
  transform: translateY(-1px);
}

.glyph-action-button:active {
  transform: translateY(0);
}

.glyph-action-button svg {
  width: 14px;
  height: 14px;
  stroke: currentColor;
}

.glyph-action-button.save.success {
  background-color: #2d5a2d;
  border-color: #3a6a3a;
  color: #8fc98f;
}

.glyph-action-button.save.success svg {
  animation: checkmark 0.3s ease-out;
}

@keyframes checkmark {
  0% { transform: scale(0.8); opacity: 0; }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); opacity: 1; }
}

.glyph-editor-footer {
  padding: 12px 0 0 0;
  text-align: center;
  color: #666;
  font-size: 12px;
  border-top: 1px solid #2a2a2a;
  margin-top: 10px;
}

.hint {
  opacity: 0.8;
}

/* Scrollbar styling */
.glyph-table-container::-webkit-scrollbar {
  width: 10px;
}

.glyph-table-container::-webkit-scrollbar-track {
  background: #1a1a1a;
  border-radius: 5px;
}

.glyph-table-container::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 5px;
  transition: background 0.2s;
}

.glyph-table-container::-webkit-scrollbar-thumb:hover {
  background: #444;
}

#glyph-editor-modal {
  position: relative;
}


/* Responsive adjustments */
@media (max-width: 768px) {
  .glyph-editor-content {
    width: 95%;
    max-height: 90vh;
  }

  .glyph-editor-controls {
    flex-direction: column;
    align-items: stretch;
  }

  .glyph-search {
    min-width: auto;
  }

  .glyph-button {
    justify-content: center;
  }
}
`

// Export the associations for use in other parts of the application
export function getGlyphAssociations(): Array<string> {
  return [...glyphAssociations]
}
