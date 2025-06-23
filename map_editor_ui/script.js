document.addEventListener('DOMContentLoaded', () => {
    // Feature detection for required browser features
    const unsupportedFeatures = [];
    
    if (!window.FileReader) unsupportedFeatures.push("FileReader API");
    if (!navigator.clipboard || !navigator.clipboard.writeText) unsupportedFeatures.push("Clipboard API");
    if (!window.HTMLCanvasElement) unsupportedFeatures.push("Canvas");
    
    if (unsupportedFeatures.length > 0) {
        console.warn("Browser missing required features:", unsupportedFeatures);
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            const loadingText = loadingOverlay.querySelector('p');
            if (loadingText) {
                loadingText.innerHTML = `<strong>Warning:</strong> Your browser may not support all features:<br>${unsupportedFeatures.join(", ")}<br><br>The app will try to run with limited functionality.`;
            }
        }
    }
    
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');

    const widthInput = document.getElementById('mapWidth');
    const heightInput = document.getElementById('mapHeight');
    const createGridBtn = document.getElementById('createGridBtn');
    const asciiPreviewTextarea = document.getElementById('asciiPreview');
    const copyAsciiBtn = document.getElementById('copyAsciiBtn');
    const loadAsciiBtn = document.getElementById('loadAsciiBtn');
    const copyStatusMessage = document.getElementById('copyStatusMessage');
    const entitySelector = document.getElementById('entitySelector');
    const toggleAsciiEditBtn = document.getElementById('toggleAsciiEditBtn');

    let gridWidth = parseInt(widthInput.value);
    let gridHeight = parseInt(heightInput.value);
    const cellSize = 20; // Size of each cell in pixels
    let grid = []; // Stores the internal drawable map
    let selectedEntity = 'wall'; // Default selected entity
    let asciiEditMode = false;
    let asciiEditorInput = null;
    let editingCell = null;

    const asciiSymbols = {
        empty: ' ',
        wall: '#',
        'agent.agent': 'A',
        mine: 'g',
        generator: 'c',
        altar: 'a',
        armory: 'r',
        lasery: 'l',
        lab: 'b',
        factory: 'f',
        temple: 't',
        'agent.team_1': 'Q',
        'agent.team_2': 'E',
        'agent.team_3': 'R',
        'agent.team_4': 'T'
    };
    const asciiToObject = {};
    for (const [obj, ch] of Object.entries(asciiSymbols)) {
        asciiToObject[ch] = obj;
    }
    asciiToObject['🧱'] = 'wall';
    asciiToObject['⚙'] = 'generator';
    asciiToObject['⛩'] = 'altar';
    asciiToObject['🏭'] = 'factory';
    asciiToObject['🔬'] = 'lab';
    asciiToObject['🏰'] = 'temple';
    // Support old wall character
    asciiToObject['W'] = 'wall';

    // Stores final sources for images (paths or data URLs)
    const objectIcons = {
        wall: 'assets/objects/wall.png',
        'agent.agent': 'assets/objects/agent.png', // Base agent icon, team icons will use SVGs
        mine: 'assets/objects/mine.png',
        generator: 'assets/objects/generator.png',
        altar: 'assets/objects/altar.png',
        armory: 'assets/objects/armory.png',
        lasery: 'assets/objects/lasery.png',
        lab: 'assets/objects/lab.png',
        factory: 'assets/objects/factory.png',
        temple: 'assets/objects/temple.png'
        // Team-specific icon sources will be generated as data URLs by prepareTeamIcons
    };

    const teamColors = {
        'agent.team_1': '#d9534f', // red
        'agent.team_2': '#0275d8', // blue
        'agent.team_3': '#5cb85c', // green
        'agent.team_4': '#f0ad4e'  // orange
    };

    // Stores HTMLImageElement objects, used for drawing on canvas
    const objectImages = {};

    // Create SVG fallback for entities with missing images
    function createSvgFallback(entityName) {
        const color = entityName.startsWith('agent.team') 
            ? teamColors[entityName] || '#000000'
            : '#3498db'; // Default blue for other entities
        
        // Generate simple SVG with the first letter of the entity name
        const firstChar = entityName.split('.').pop().charAt(0).toUpperCase();
        const svgData = `
            <svg xmlns="http://www.w3.org/2000/svg" width="${cellSize}" height="${cellSize}" viewBox="0 0 ${cellSize} ${cellSize}">
                <rect width="${cellSize}" height="${cellSize}" fill="${color}" />
                <text x="${cellSize / 2}" y="${cellSize * 0.7}" font-family="sans-serif" font-size="${cellSize * 0.6}" 
                      text-anchor="middle" fill="white">${firstChar}</text>
            </svg>
        `;
        return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgData.trim())}`;
    }

    function prepareTeamIcons() {
        // Create simple colored squares for team icons using SVG data URLs
        Object.entries(teamColors).forEach(([obj, color]) => {
            const svgData = `
                <svg xmlns="http://www.w3.org/2000/svg" width="${cellSize}" height="${cellSize}">
                    <rect width="${cellSize}" height="${cellSize}" fill="${color}" />
                    <text x="${cellSize / 2}" y="${cellSize * 0.7}" font-family="sans-serif" font-size="${cellSize * 0.6}" text-anchor="middle" fill="white">T</text>
                </svg>
            `;
            const dataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgData.trim())}`;
            
            objectIcons[obj] = dataUrl; // Update the source in objectIcons
            
            const img = new Image();
            img.src = dataUrl; // Data URLs load synchronously
            objectImages[obj] = img; // Add the Image element to objectImages
        });
    }

    let mouseButtonPressed = null;
    let lastProcessedCell = { row: null, col: null };
    let isConfirmingReset = false;
    let resetTimeoutId = null;
    let originalCreateBtnText = ''; // Will be set after DOM content is loaded
    let originalCreateBtnBgColor = ''; // Will be set after DOM content is loaded


    function initializeGrid(width, height) {
        gridWidth = width;
        gridHeight = height;
        grid = Array(gridHeight)
            .fill(null)
            .map(() => Array(gridWidth).fill(asciiSymbols.empty));

        canvas.width = (gridWidth + 2) * cellSize;
        canvas.height = (gridHeight + 2) * cellSize;

        drawCanvas();
        updateAsciiPreview();
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let r = 0; r < gridHeight + 2; r++) {
            for (let c = 0; c < gridWidth + 2; c++) {
                const isBorderCell = (r === 0 || r === gridHeight + 1 || c === 0 || c === gridWidth + 1);
                const char = isBorderCell ? asciiSymbols.wall : grid[r - 1][c - 1];
                const obj = asciiToObject[char] || 'empty';
                const imgToDraw = objectImages[obj];

                // Background fill regardless of what we're drawing
                ctx.fillStyle = '#e0e0e0';
                ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);

                if (asciiEditMode) {
                    if (char !== asciiSymbols.empty) {
                        ctx.fillStyle = '#000';
                        ctx.font = '16px monospace';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(char, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
                    }
                } else if (obj !== 'empty' && imgToDraw && imgToDraw.complete && imgToDraw.naturalWidth !== 0) {
                    // Draw the image if it's loaded properly
                    ctx.drawImage(imgToDraw, c * cellSize, r * cellSize, cellSize, cellSize);
                } else if (char !== asciiSymbols.empty && obj === 'empty') { 
                    // If it was an unknown char, draw it
                    ctx.fillStyle = '#000';
                    ctx.font = '16px monospace';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(char, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
                } else if (obj !== 'empty') {
                    // Draw a fallback representation for image that didn't load properly
                    // Get the first character of the object type
                    const objName = obj.split('.').pop();
                    const label = objName.charAt(0).toUpperCase();
                    
                    ctx.fillStyle = '#888';
                    ctx.font = '16px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(label, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
                }
            }
        }

        // Grid lines
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        for (let r = 0; r <= gridHeight + 2; r++) {
            ctx.beginPath();
            ctx.moveTo(0, r * cellSize);
            ctx.lineTo(canvas.width, r * cellSize);
            ctx.stroke();
        }
        for (let c = 0; c <= gridWidth + 2; c++) {
            ctx.beginPath();
            ctx.moveTo(c * cellSize, 0);
            ctx.lineTo(c * cellSize, canvas.height);
            ctx.stroke();
        }
    }

    function updateAsciiPreview() {
        let ascii = '';
        const wallChar = asciiSymbols.wall;
        ascii += wallChar.repeat(gridWidth + 2) + '\n';
        for (let r = 0; r < gridHeight; r++) {
            ascii += wallChar;
            for (let c = 0; c < gridWidth; c++) {
                ascii += grid[r][c];
            }
            ascii += wallChar + '\n';
        }
        ascii += wallChar.repeat(gridWidth + 2) + '\n';
        asciiPreviewTextarea.value = ascii.trim();
        asciiPreviewTextarea.style.height = 'auto';
        asciiPreviewTextarea.style.height = (asciiPreviewTextarea.scrollHeight) + 'px';
    }

    function loadFromAscii(text) {
        const lines = text.trim().split(/\r?\n/).filter(l => l.length);
        if (lines.length < 3) return false;
        const innerWidth = lines[0].length - 2;
        const innerHeight = lines.length - 2;
        if (innerWidth <= 0 || innerHeight <= 0) return false;
        for (const line of lines) {
            if (line.length !== innerWidth + 2) return false;
        }
        initializeGrid(innerWidth, innerHeight); // This will set gridWidth, gridHeight
        for (let r = 0; r < innerHeight; r++) {
            const row = lines[r + 1];
            for (let c = 0; c < innerWidth; c++) {
                if (r < gridHeight && c < gridWidth) { // Ensure we don't write out of bounds
                    grid[r][c] = row[c + 1];
                }
            }
        }
        drawCanvas();
        updateAsciiPreview();
        return true;
    }

    function getMouseGridPos(event) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        const col = Math.floor(mouseX / cellSize);
        const row = Math.floor(mouseY / cellSize);
        return { row: row - 1, col: col - 1 };
    }

    function applyToCell(row, col, cellValueToSet) {
        if (row >= 0 && row < gridHeight && col >= 0 && col < gridWidth) {
            if (grid[row][col] !== cellValueToSet) {
                grid[row][col] = cellValueToSet;
                return true;
            }
        }
        return false;
    }

    function drawLine(r0, c0, r1, c1, cellValueToSet) {
        let changed = false;
        const dr = Math.abs(r1 - r0);
        const dc = Math.abs(c1 - c0);
        const sr = (r0 < r1) ? 1 : -1;
        const sc = (c0 < c1) ? 1 : -1;
        let err = dr - dc;
        let r = r0;
        let c = c0;
        while (true) {
            if (applyToCell(r, c, cellValueToSet)) changed = true;
            if (r === r1 && c === c1) break;
            const e2 = 2 * err;
            if (e2 > -dc) { err -= dc; r += sr; }
            if (e2 < dr) { err += dr; c += sc; }
        }
        return changed;
    }

    function openAsciiEditor(row, col) {
        closeAsciiEditor(false);
        editingCell = { row, col };
        if (!asciiEditorInput) {
            asciiEditorInput = document.createElement('input');
            asciiEditorInput.maxLength = 1;
            asciiEditorInput.id = 'ascii-editor-input';
            Object.assign(asciiEditorInput.style, {
                position: 'absolute', textAlign: 'center', fontFamily: 'monospace',
                padding: '0', margin: '0', border: '1px solid #666', boxSizing: 'border-box'
            });
            asciiEditorInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault(); // Prevent default to avoid submitting forms
                    closeAsciiEditor(true);
                }
                else if (e.key === 'Escape') {
                    e.preventDefault(); // Prevent default to avoid browser actions
                    closeAsciiEditor(false);
                }
            });
            // Use a timeout to avoid race conditions with other event handlers
            asciiEditorInput.addEventListener('blur', () => {
                // Small delay to prevent race conditions with Enter/Escape key handlers
                setTimeout(() => {
                    if (editingCell) { // Only close if still editing
                        closeAsciiEditor(true);
                    }
                }, 10);
            });
        }
        Object.assign(asciiEditorInput.style, {
            width: cellSize + 'px', height: cellSize + 'px',
            left: canvas.offsetLeft + (col + 1) * cellSize + 'px',
            top: canvas.offsetTop + (row + 1) * cellSize + 'px'
        });
        asciiEditorInput.value = grid[row][col];
        
        // Remove any existing instance first to avoid duplicates
        const existingInput = document.getElementById('ascii-editor-input');
        if (existingInput && existingInput !== asciiEditorInput) {
            existingInput.remove();
        }
        
        document.body.appendChild(asciiEditorInput);
        asciiEditorInput.focus();
        asciiEditorInput.select();
    }

    function closeAsciiEditor(commit) {
        // Use a try-catch to handle any potential DOM errors
        try {
            // Guard against calls when no edit is active or input element is missing.
            if (!editingCell || !asciiEditorInput) {
                // If asciiEditorInput still exists and editingCell is null (e.g. a late blur event after Enter press),
                // ensure it's removed if it's still in the DOM to clean up.
                if (asciiEditorInput && document.getElementById('ascii-editor-input')) {
                    try {
                        document.getElementById('ascii-editor-input').remove();
                    } catch (e) {
                        // Silent fail
                    }
                }
                return;
            }

            const currentEditingCell = editingCell; // Capture before nulling
            const currentValue = asciiEditorInput.value; // Capture value before input might be removed

            editingCell = null; // Mark editing as finished immediately to prevent re-entrancy from subsequent events

            if (commit) {
                const ch = currentValue ? currentValue[0] : asciiSymbols.empty;
                if (currentEditingCell.row < gridHeight && currentEditingCell.col < gridWidth) { // Check bounds
                    grid[currentEditingCell.row][currentEditingCell.col] = ch;
                }
                updateAsciiPreview();
                drawCanvas();
            }

            // Use safer DOM query to find and remove the element
            const inputElement = document.getElementById('ascii-editor-input');
            if (inputElement) {
                inputElement.remove();
            }
            // Do NOT null out asciiEditorInput itself, as it's a shared DOM element that gets reused.
        } catch (err) {
            // Try to recover by resetting state
            editingCell = null;
            // Try one more time to remove the element if it exists
            try {
                const inputElement = document.getElementById('ascii-editor-input');
                if (inputElement) {
                    inputElement.remove();
                }
            } catch (e) {
                // Last resort, just ignore
            }
        }
    }

    function handleInteraction(event) {
        if (asciiEditMode) return;
        const { row, col } = getMouseGridPos(event);
        let needsRedraw = false;
        let cellValueToSet;

        if (!asciiSymbols[selectedEntity]) {
            console.warn(`Selected entity '${selectedEntity}' has no ASCII symbol. Defaulting to empty.`);
            cellValueToSet = asciiSymbols.empty;
        } else {
             cellValueToSet = (mouseButtonPressed === 0) ? asciiSymbols[selectedEntity] : asciiSymbols.empty;
        }
       
        if (mouseButtonPressed === null && (event.type === 'mousedown' || event.type === 'contextmenu')) {
            // This logic seems flawed, mouseButtonPressed is set before calling handleInteraction.
            // Let's rely on the mouseButtonPressed state set by event listeners.
            return;
        }
        
        if (mouseButtonPressed !== 0 && mouseButtonPressed !== 2) return; // Only handle left/right press


        if (event.type === 'mousedown' || event.type === 'contextmenu') {
            if (applyToCell(row, col, cellValueToSet)) needsRedraw = true;
            lastProcessedCell = { row, col };
        } else if (event.type === 'mousemove' && mouseButtonPressed !== null) {
            if (lastProcessedCell.row !== null && (lastProcessedCell.row !== row || lastProcessedCell.col !== col)) {
                if (drawLine(lastProcessedCell.row, lastProcessedCell.col, row, col, cellValueToSet)) needsRedraw = true;
                lastProcessedCell = { row, col };
            } else if (lastProcessedCell.row === null) {
                if (applyToCell(row, col, cellValueToSet)) needsRedraw = true;
                lastProcessedCell = { row, col };
            }
        }
        if (needsRedraw) {
            drawCanvas();
            updateAsciiPreview();
        }
    }

    canvas.addEventListener('mousedown', (event) => {
        if (asciiEditMode) return;
        if (event.button === 0) { mouseButtonPressed = 0; handleInteraction(event); }
    });
    canvas.addEventListener('contextmenu', (event) => {
        if (asciiEditMode) return;
        event.preventDefault(); mouseButtonPressed = 2; handleInteraction(event);
    });
    canvas.addEventListener('mousemove', (event) => {
        if (asciiEditMode || mouseButtonPressed === null) return;
        handleInteraction(event);
    });
    canvas.addEventListener('mouseup', () => {
        mouseButtonPressed = null; lastProcessedCell = { row: null, col: null };
    });
    canvas.addEventListener('mouseleave', () => {
        mouseButtonPressed = null; lastProcessedCell = { row: null, col: null };
    });
    canvas.addEventListener('click', (event) => {
        if (!asciiEditMode) return;
        const { row, col } = getMouseGridPos(event);
        if (row >= 0 && row < gridHeight && col >= 0 && col < gridWidth) {
            openAsciiEditor(row, col);
        }
    });

    copyAsciiBtn.addEventListener('click', () => {
        // Show immediate feedback before the copy operation completes
        const originalText = copyAsciiBtn.textContent;
        copyAsciiBtn.textContent = 'Copying...';
        copyStatusMessage.textContent = '';
        copyStatusMessage.style.visibility = 'hidden';
        
        // Fallback copy method using selection
        function fallbackCopy() {
            try {
                asciiPreviewTextarea.select();
                const success = document.execCommand('copy');
                
                if (success) {
                    copyStatusMessage.textContent = 'Copied!'; 
                    copyStatusMessage.style.color = '#28a745';
                } else {
                    copyStatusMessage.textContent = 'Please press Ctrl+C to copy'; 
                    copyStatusMessage.style.color = '#f0ad4e';
                }
                
                copyStatusMessage.style.visibility = 'visible';
                copyAsciiBtn.textContent = originalText;
                
                setTimeout(() => { 
                    copyStatusMessage.style.visibility = 'hidden';
                    copyStatusMessage.style.color = '#28a745';
                }, 3000);
            } catch (err) {
                copyStatusMessage.textContent = 'Failed to copy. Please select and copy manually.'; 
                copyStatusMessage.style.color = 'red';
                copyStatusMessage.style.visibility = 'visible';
                copyAsciiBtn.textContent = originalText;
                
                console.error('Fallback copy failed: ', err);
            }
        }
        
        // Try using the clipboard API first
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(asciiPreviewTextarea.value)
                .then(() => {
                    copyStatusMessage.textContent = 'Copied!'; 
                    copyStatusMessage.style.color = '#28a745';
                    copyStatusMessage.style.visibility = 'visible';
                    copyAsciiBtn.textContent = 'Copied!';
                    
                    setTimeout(() => { 
                        copyStatusMessage.style.visibility = 'hidden';
                        copyAsciiBtn.textContent = originalText;
                    }, 2000);
                })
                .catch(err => {
                    console.error('Failed to copy: ', err);
                    fallbackCopy();
                });
        } else {
            // Fallback for browsers without clipboard API
            fallbackCopy();
        }
    });

    createGridBtn.addEventListener('click', () => {
        if (isConfirmingReset) {
            clearTimeout(resetTimeoutId); document.removeEventListener('click', handleOutsideClickForReset, true);
            resetCreateButtonState(); isConfirmingReset = false;
            const newWidth = parseInt(widthInput.value); const newHeight = parseInt(heightInput.value);
            if (newWidth >= 3 && newHeight >= 3 && newWidth <= 100 && newHeight <= 100) {
                initializeGrid(newWidth, newHeight);
            } else { alert('Width/Height must be 3-100.'); }
        } else {
            isConfirmingReset = true; createGridBtn.textContent = 'Are you sure?'; createGridBtn.style.backgroundColor = 'red';
            resetTimeoutId = setTimeout(() => {
                resetCreateButtonState(); isConfirmingReset = false;
                document.removeEventListener('click', handleOutsideClickForReset, true);
            }, 5000);
            document.addEventListener('click', handleOutsideClickForReset, true);
        }
    });

    function resetCreateButtonState() {
        createGridBtn.textContent = originalCreateBtnText;
        createGridBtn.style.backgroundColor = originalCreateBtnBgColor;
        if (resetTimeoutId) { clearTimeout(resetTimeoutId); resetTimeoutId = null; }
    }

    function handleOutsideClickForReset(event) {
        if (isConfirmingReset && event.target !== createGridBtn) {
            resetCreateButtonState(); isConfirmingReset = false;
            document.removeEventListener('click', handleOutsideClickForReset, true);
        }
    }

    loadAsciiBtn.addEventListener('click', () => {
        if (!loadFromAscii(asciiPreviewTextarea.value)) alert('Invalid ASCII map format.');
    });

    toggleAsciiEditBtn.addEventListener('click', () => {
        asciiEditMode = !asciiEditMode;
        toggleAsciiEditBtn.classList.toggle('active', asciiEditMode);
        if (!asciiEditMode && editingCell) closeAsciiEditor(true); // Commit changes if exiting ASCII mode
        drawCanvas(); // Redraw to switch between char/image mode
    });
    
    function createEntityButtons() {
        entitySelector.innerHTML = ''; // Clear previous buttons
        Object.keys(objectIcons).forEach((objKey) => { // Iterate over keys of objectIcons
            const srcOrDataUrl = objectIcons[objKey];
            const btn = document.createElement('button');
            btn.className = 'entity-btn';
            btn.dataset.entity = objKey;
            
            // Create a user-friendly name for the tooltip
            const displayName = objKey.split('.').pop() // Get the last part after any dots
                                     .replace(/([A-Z])/g, ' $1') // Add space before capitals (camelCase to words)
                                     .toLowerCase() // Convert to lowercase
                                     .split(' ') // Split into words
                                     .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // Capitalize first letter
                                     .join(' '); // Join back with spaces
                                     
            // Add tooltip
            btn.title = displayName;
            
            const imgElementForButton = document.createElement('img');
            imgElementForButton.src = srcOrDataUrl;
            imgElementForButton.alt = displayName;
            imgElementForButton.style.width = cellSize + 'px';
            imgElementForButton.style.height = cellSize + 'px';
            imgElementForButton.onerror = () => { // Add error handler for button images
                imgElementForButton.alt = `${displayName} (failed to load)`;
                imgElementForButton.src = ''; // Clear src to prevent broken image icon
                // Optionally, display placeholder text or style differently
                btn.title = `${displayName} (image failed to load)`;
            };
            btn.appendChild(imgElementForButton);
            btn.addEventListener('click', () => {
                selectedEntity = objKey;
                document.querySelectorAll('.entity-btn.selected').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
            });
            entitySelector.appendChild(btn);
            if (objKey === selectedEntity) {
                btn.classList.add('selected');
            }
        });

        // Fallback: if current selectedEntity didn't get a button selected (e.g. it's not in objectIcons)
        // or if nothing is selected, select the first available button.
        if (!document.querySelector('.entity-btn.selected')) {
            const firstButton = entitySelector.querySelector('.entity-btn');
            if (firstButton) {
                firstButton.classList.add('selected');
                selectedEntity = firstButton.dataset.entity;
            }
        }
    }

    function performInitialization() {
        originalCreateBtnText = createGridBtn.textContent; // Store original button text
        originalCreateBtnBgColor = window.getComputedStyle(createGridBtn).backgroundColor; // Store original bg color

        // Ensure selectedEntity is valid and has a loaded image, or default
        if (!objectIcons[selectedEntity]) { // Check against objectIcons first
             const firstAvailableKeyFromIcons = Object.keys(objectIcons)[0];
             if (firstAvailableKeyFromIcons) {
                 selectedEntity = firstAvailableKeyFromIcons;
             } else {
                 console.error("No entities defined in objectIcons. Cannot select an entity.");
                 selectedEntity = 'empty'; // Fallback to a harmless pseudo-entity
                 if(!asciiSymbols[selectedEntity]) asciiSymbols[selectedEntity] = ' '; // Ensure empty exists
             }
        }
        
        createEntityButtons(); // Create buttons based on final objectIcons and selectedEntity
        initializeGrid(gridWidth, gridHeight); // Initialize and draw the grid
        
        // Hide loading overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.opacity = '0';
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
            }, 300); // Match transition duration
        }
    }

    function loadAndInitialize() {
        prepareTeamIcons(); // Prepares team SVGs and updates objectIcons/objectImages

        let imagesToLoadCount = 0;
        let imagesLoadedCount = 0;
        const pathBasedImageKeys = [];

        Object.keys(objectIcons).forEach(objKey => {
            const src = objectIcons[objKey];
            // Check if it's a path (not a data URL) and not already processed as a team icon's Image object
            if (typeof src === 'string' && !src.startsWith('data:')) {
                if (!objectImages[objKey] || !(objectImages[objKey] instanceof HTMLImageElement)) {
                    pathBasedImageKeys.push(objKey);
                    imagesToLoadCount++;
                }
            } else if (src.startsWith('data:') && !objectImages[objKey]) {
                 // This case handles data URLs that might not have been added to objectImages
                 // by prepareTeamIcons (though current prepareTeamIcons does add them).
                 // This is defensive. Data URLs are considered loaded.
                const img = new Image();
                img.src = src;
                objectImages[objKey] = img;
            }
        });

        if (imagesToLoadCount === 0) {
            performInitialization(); // All images are data URLs or no path images to load
            return;
        }

        // Set a timeout to continue initialization even if images take too long to load
        const loadTimeoutId = setTimeout(() => {
            console.warn("Some images are taking too long to load, continuing with initialization");
            if (imagesLoadedCount < imagesToLoadCount) {
                // Create fallbacks for any images that haven't loaded yet
                pathBasedImageKeys.forEach(objKey => {
                    if (!objectImages[objKey] || !objectImages[objKey].complete) {
                        console.log(`Creating fallback for timed-out image: ${objKey}`);
                        const fallbackSvg = createSvgFallback(objKey);
                        const fallbackImg = new Image();
                        fallbackImg.src = fallbackSvg;
                        objectImages[objKey] = fallbackImg;
                        objectIcons[objKey] = fallbackSvg;
                    }
                });
                performInitialization();
            }
        }, 3000); // 3 second timeout

        pathBasedImageKeys.forEach(objKey => {
            const img = new Image();
            img.onload = () => {
                imagesLoadedCount++;
                if (imagesLoadedCount === imagesToLoadCount) {
                    clearTimeout(loadTimeoutId);
                    performInitialization();
                }
            };
            img.onerror = () => {
                console.error(`Failed to load image: ${objKey} from ${objectIcons[objKey]}`);
                
                // Create fallback SVG image if loading fails
                const fallbackSvg = createSvgFallback(objKey);
                
                // Use the fallback SVG instead
                const fallbackImg = new Image();
                fallbackImg.src = fallbackSvg;
                objectImages[objKey] = fallbackImg;
                objectIcons[objKey] = fallbackSvg; // Update objectIcons as well
                
                imagesLoadedCount++; // Count as "resolved" to not hang
                // drawCanvas needs to handle this (e.g., by drawing a placeholder).
                if (imagesLoadedCount === imagesToLoadCount) {
                    clearTimeout(loadTimeoutId);
                    performInitialization();
                }
            };
            img.src = objectIcons[objKey];
            objectImages[objKey] = img;
        });
    }

    // Initial setup call
    loadAndInitialize();
});