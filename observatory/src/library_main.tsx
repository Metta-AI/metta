import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Library } from './Library'
import { BrowserRouter } from 'react-router-dom'
import './index.css'

createRoot(document.getElementById('library-root')!).render(
    <StrictMode>
        <BrowserRouter>
            <Library repo={undefined} />
        </BrowserRouter>
    </StrictMode>
)
