"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

// MathJax type declarations
declare global {
  interface Window {
    MathJax?: {
      tex: {
        inlineMath: string[][]
        displayMath: string[][]
        processEscapes: boolean
        processEnvironments: boolean
      }
      options: {
        skipHtmlTags: string[]
      }
      typesetPromise?: (elements: HTMLElement[]) => Promise<void>
    }
  }
}

interface MathJaxContextType {
  mathJaxLoaded: boolean;
  renderMath: (element: HTMLElement) => Promise<void>;
}

const MathJaxContext = createContext<MathJaxContextType | null>(null);

export function useMathJax() {
  const context = useContext(MathJaxContext);
  if (!context) {
    throw new Error('useMathJax must be used within a MathJaxProvider');
  }
  return context;
}

interface MathJaxProviderProps {
  children: ReactNode;
}

/**
 * MathJaxProvider Component
 * 
 * Provides MathJax functionality throughout the application:
 * - Loads MathJax from CDN
 * - Configures MathJax for LaTeX rendering
 * - Provides context for components to render mathematical content
 * - Handles both inline ($...$) and display ($$...$$) math
 */
export function MathJaxProvider({ children }: MathJaxProviderProps) {
  const [mathJaxLoaded, setMathJaxLoaded] = useState(false);

  useEffect(() => {
    const loadMathJax = async () => {
      if (typeof window !== 'undefined' && !window.MathJax) {
        // Configure MathJax before loading
        (window as any).MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true
          },
          options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
          }
        };

        // Load MathJax dynamically
        const mathJaxScript = document.createElement('script');
        mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
        mathJaxScript.async = true;
        mathJaxScript.onload = () => {
          // Wait a bit for MathJax to fully initialize
          setTimeout(() => {
            if (window.MathJax) {
              setMathJaxLoaded(true);
            }
          }, 200);
        };
        document.head.appendChild(mathJaxScript);
      } else if (window.MathJax) {
        setMathJaxLoaded(true);
      }
    };

    loadMathJax();
  }, []);

  const renderMath = async (element: HTMLElement) => {
    if (mathJaxLoaded && window.MathJax?.typesetPromise) {
      try {
        await window.MathJax.typesetPromise([element]);
      } catch (err) {
        console.error('MathJax rendering error:', err);
      }
    }
  };

  return (
    <MathJaxContext.Provider value={{ mathJaxLoaded, renderMath }}>
      {children}
    </MathJaxContext.Provider>
  );
} 