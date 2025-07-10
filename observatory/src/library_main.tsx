import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Library } from './Library'
import './index.css'

createRoot(document.getElementById('library-root')!).render(
    <StrictMode>
        <Library repo={undefined} />
    </StrictMode>
)
