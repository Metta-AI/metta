import { useState, useEffect, useRef } from 'react'

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

// CSS for feed page with new design system
const FEED_CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow-x: hidden;
}

.feed-container {
  display: flex;
  min-height: calc(100vh - 60px);
  font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  margin-top: 60px;
}

.feed-sidebar {
  width: 240px;
  background: #ffffff;
  border-right: 1px solid #e5e7eb;
  position: fixed;
  top: 60px;
  left: 0;
  height: calc(100vh - 60px);
  display: flex;
  flex-direction: column;
  z-index: 10;
  overflow-y: auto;
}

.sidebar-nav {
  flex: 1;
  padding: 16px 0;
}

.sidebar-cta {
  padding: 16px;
  border-top: 1px solid #e5e7eb;
  flex-shrink: 0;
}

.feed-main {
  flex: 1;
  background: #ffffff;
  margin-left: 240px;
  min-height: calc(100vh - 60px);
  padding-bottom: 40px;
  overflow-y: auto;
}

/* Post Composer */
.post-composer {
  background: #ffffff;
  border-bottom: 1px solid #e5e7eb;
  padding: 16px 24px;
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.composer-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #2D5BFF;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 14px;
  flex-shrink: 0;
}

.composer-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.composer-textarea {
  width: 100%;
  min-height: 60px;
  max-height: 120px;
  border: none;
  resize: none;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  outline: none;
  padding: 0;
}

.composer-textarea::placeholder {
  color: #9ca3af;
}

.composer-actions {
  display: flex;
  justify-content: flex-end;
}

.post-button {
  background: #2D5BFF;
  color: white;
  border: none;
  border-radius: 20px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.post-button:disabled {
  background: #d1d5db;
  cursor: not-allowed;
}

.post-button:not(:disabled):hover {
  background: #1e40af;
}

/* Feed Items */
.feed-list {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.feed-item {
  padding: 16px 24px;
  border-bottom: 1px solid #e5e7eb;
}

.feed-item.user-post {
  background: #EDF1FF;
  border-radius: 8px;
  margin: 0 24px 0 24px;
  border-bottom: none;
}

.feed-item.paper-post {
  background: #FFF7E8;
  border-radius: 8px;
  margin: 0 24px 0 24px;
  border-bottom: none;
}

.feed-item.pure-paper {
  background: #ffffff;
  border-radius: 8px;
  margin: 0 24px 0 24px;
  border-bottom: none;
}

.post-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.post-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #2D5BFF;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 12px;
}

.post-user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.post-name {
  font-weight: 600;
  color: #111827;
  font-size: 14px;
}

.post-time {
  color: #6b7280;
  font-size: 13px;
}

.post-content {
  color: #111827;
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 16px;
}

.post-content .math {
  display: inline-block;
  margin: 0 2px;
}

.post-content .math-display {
  display: block;
  text-align: center;
  margin: 12px 0;
}

.post-actions {
  display: flex;
  gap: 48px;
  color: #6b7280;
  font-size: 13px;
}

.post-action {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: color 0.2s ease;
}

.post-action:hover {
  color: #2D5BFF;
}

/* PaperCard */
.paper-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.paper-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.paper-avatar {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #f59e0b;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 10px;
}

.paper-author {
  font-size: 14px;
  font-weight: 500;
  color: #111827;
}

.paper-title {
  font-size: 18px;
  font-weight: 600;
  color: #111827;
  display: flex;
  align-items: center;
  gap: 8px;
  line-height: 1.4;
}

.paper-title svg {
  width: 16px;
  height: 16px;
  color: #6b7280;
}

.paper-summary {
  border-left: 4px solid #2D5BFF;
  background: #F2F6FF;
  padding: 8px 12px;
  font-size: 13px;
  line-height: 1.5;
  color: #374151;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.paper-abstract {
  font-size: 14px;
  line-height: 1.6;
  color: #374151;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.paper-abstract.expanded {
  display: block;
  -webkit-line-clamp: unset;
}

.paper-badge {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: #6b7280;
}

.paper-badge-left {
  background: #f3f4f6;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
}

.paper-badge-right {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
}

.paper-badge-right svg {
  width: 14px;
  height: 14px;
  color: #ef4444;
}

.show-more-btn {
  color: #2D5BFF;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  background: none;
  border: none;
  padding: 0;
  margin-top: 4px;
}

.show-more-btn:hover {
  text-decoration: underline;
}



.nav-item {
  display: flex;
  align-items: center;
  height: 40px;
  padding: 0 12px;
  color: #6b7280;
  text-decoration: none;
  font-weight: 500;
  font-size: 14px;
  transition: all 0.2s ease;
  border-radius: 20px;
  margin: 0 12px 2px 12px;
  cursor: pointer;
  gap: 8px;
}

.nav-item:hover {
  background-color: #f3f4f6;
  color: #111827;
}

.nav-item.active {
  background-color: rgba(45, 91, 255, 0.1);
  color: #2D5BFF;
  font-weight: 600;
}

.nav-item span {
  display: flex;
  align-items: center;
}

.nav-item span svg {
  width: 20px;
  height: 20px;
}
}

.post-action {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: color 0.2s ease;
  padding: 4px 8px;
  border-radius: 6px;
}

.post-action:hover {
  color: #2D5BFF;
  background-color: #f3f4f6;
}

.post-action span {
  display: flex;
  align-items: center;
}

.post-action span svg {
  width: 16px;
  height: 16px;
}

.compose-button {
  background: #2D5BFF;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 12px 20px;
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  width: calc(100% - 24px);
  margin: 0 12px 20px 12px;
  transition: background-color 0.2s ease;
}

.compose-button:hover {
  background: #1e40af;
}

.cta-button {
  width: 100%;
  height: 40px;
  background-color: #2D5BFF;
  color: #ffffff;
  border: none;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cta-button:hover {
  background-color: #1e40af;
  transform: translateY(-1px);
}



.paper-preview {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
  margin: 12px 0;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.paper-title {
  font-weight: 600;
  color: #111827;
  font-size: 14px;
  margin-bottom: 4px;
}

.paper-authors {
  color: #6b7280;
  font-size: 12px;
  margin-bottom: 8px;
}

.paper-abstract {
  color: #374151;
  font-size: 13px;
  line-height: 1.5;
}

/* Custom scrollbar for sidebar */
.feed-sidebar::-webkit-scrollbar {
  width: 6px;
}

.feed-sidebar::-webkit-scrollbar-track {
  background: #f1f5f9;
}

.feed-sidebar::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

.feed-sidebar::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
`

interface LibraryProps {
  repo: unknown // Using unknown for now since this is a dummy component
}

export function Library({ repo: _repo }: LibraryProps) {
  const [activeNav, setActiveNav] = useState('feed')
  const [mathJaxLoaded, setMathJaxLoaded] = useState(false)
  const postsRef = useRef<HTMLDivElement>(null)
  const [composerText, setComposerText] = useState('')
  const [expandedAbstracts, setExpandedAbstracts] = useState<Set<number>>(new Set())

  // Initialize MathJax
  useEffect(() => {
    const loadMathJax = async () => {
      if (typeof window !== 'undefined' && !window.MathJax) {
        // Configure MathJax before loading
        window.MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true
          },
          options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
          }
        }
        
        // Load MathJax dynamically
        const mathJaxScript = document.createElement('script')
        mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
        mathJaxScript.async = true
        mathJaxScript.onload = () => {
          // Wait a bit for MathJax to fully initialize
          setTimeout(() => {
            if (window.MathJax) {
              setMathJaxLoaded(true)
            }
          }, 200)
        }
        document.head.appendChild(mathJaxScript)
      } else if (window.MathJax) {
        setMathJaxLoaded(true)
      }
    }

    loadMathJax()
  }, [])

  // Render MathJax when posts change
  useEffect(() => {
    if (mathJaxLoaded && postsRef.current) {
      // Force MathJax to reprocess the content with multiple attempts
      const renderMath = () => {
        console.log('Attempting to render MathJax...')
        console.log('MathJax available:', !!window.MathJax)
        console.log('typesetPromise available:', !!window.MathJax?.typesetPromise)
        console.log('Posts ref:', postsRef.current)
        
        if (window.MathJax?.typesetPromise) {
          window.MathJax.typesetPromise([postsRef.current!]).then(() => {
            console.log('MathJax rendering completed successfully')
          }).catch((err: any) => {
            console.error('MathJax error:', err)
          })
        }
      }
      
      // Try immediately
      renderMath()
      
      // Try again after a short delay to ensure DOM is ready
      setTimeout(renderMath, 100)
      
      // Try one more time after a longer delay for Safari
      setTimeout(renderMath, 500)
    }
  }, [mathJaxLoaded])

  const handlePostSubmit = () => {
    if (composerText.trim()) {
      // In a real app, this would submit to the backend
      console.log('Posting:', composerText)
      setComposerText('')
    }
  }

  const toggleAbstract = (postId: number) => {
    const newExpanded = new Set(expandedAbstracts)
    if (newExpanded.has(postId)) {
      newExpanded.delete(postId)
    } else {
      newExpanded.add(postId)
    }
    setExpandedAbstracts(newExpanded)
  }

  // Dummy posts data with mathematical content
  const posts = [
    {
      id: 1,
      name: 'Dr. Alice Johnson',
      username: '@alicej',
      avatar: 'A',
      content: `Just published our latest work on attention mechanisms! The key insight is that self-attention can be viewed as a form of differentiable memory access. The attention weights $\\alpha_{ij}$ for query $i$ and key $j$ are computed as:

$$\\alpha_{ij} = \\frac{\\exp(e_{ij})}{\\sum_k \\exp(e_{ik})}$$

where $e_{ij} = \\frac{Q_i^T K_j}{\\sqrt{d_k}}$ is the scaled dot-product attention. This formulation allows the model to learn which parts of the input to focus on dynamically. 🧠 #AttentionMechanisms #DeepLearning

Test inline: $x^2 + y^2 = z^2$ and $\\alpha + \\beta = \\gamma$`,
      time: '2h',
      likes: 24,
      retweets: 8,
      replies: 3,
      type: 'user-post'
    },
    {
      id: 2,
      name: 'Prof. Bob Chen',
      username: '@bobchen',
      avatar: 'B',
      content: `Excited to share our new paper on reinforcement learning with continuous action spaces! We introduce a novel policy gradient method that uses the natural gradient $\\nabla_\\theta J(\\theta) = F^{-1}(\\theta) \\nabla_\\theta J(\\theta)$ where $F(\\theta)$ is the Fisher information matrix. This leads to more stable training and better sample efficiency.`,
      time: '4h',
      likes: 42,
      retweets: 15,
      replies: 7,
      type: 'paper-post',
      paper: {
        id: 'arxiv:1804.02464v3',
        title: 'Natural Policy Gradients for Continuous Control',
        author: 'Robert Chen',
        authorInitial: 'R',
        summary: 'This paper introduces a novel policy gradient method using natural gradients that achieves 40% improvement in sample efficiency on continuous control tasks. The key innovation is the use of the Fisher information matrix to compute more stable policy updates.',
        abstract: 'We present a novel policy gradient method for reinforcement learning in continuous action spaces that uses the natural gradient to improve sample efficiency and training stability. Our approach leverages the Fisher information matrix to compute more effective policy updates, leading to faster convergence and better final performance. We demonstrate the effectiveness of our method on a variety of continuous control benchmarks, showing significant improvements over standard policy gradient methods. The theoretical analysis provides insights into why natural gradients are particularly effective for policy optimization in continuous spaces.',
        citations: 127,
        url: 'https://arxiv.org/abs/1804.02464'
      }
    },
    {
      id: 3,
      name: 'Dr. Carol Williams',
      username: '@carolw',
      avatar: 'C',
      content: `Fascinating discussion about the universal approximation theorem in our seminar today! For any continuous function $f: [0,1]^n \\rightarrow \\mathbb{R}$ and $\\epsilon > 0$, there exists a neural network with one hidden layer that can approximate $f$ within $\\epsilon$ error. Formally:

$$|f(x) - \\sum_{i=1}^N \\alpha_i \\sigma(w_i^T x + b_i)| < \\epsilon$$

where $\\sigma$ is a sigmoid activation function. This theoretical result explains why neural networks are so powerful! 🧮 #NeuralNetworks #Theory`,
      time: '6h',
      likes: 67,
      retweets: 23,
      replies: 12,
      type: 'user-post'
    },
    {
      id: 4,
      type: 'pure-paper',
      paper: {
        id: 'arxiv:1706.03762',
        title: 'Attention Is All You Need',
        author: 'Ashish Vaswani',
        authorInitial: 'A',
        summary: 'This seminal paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms and has become the foundation for modern language models. The paper demonstrates that attention mechanisms alone can achieve state-of-the-art results in machine translation.',
        abstract: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show that these models are superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing with large amounts of training data.',
        citations: 45678,
        url: 'https://arxiv.org/abs/1706.03762'
      }
    },
    {
      id: 5,
      name: 'Eva Rodriguez',
      username: '@evarod',
      avatar: 'E',
      content: `Just finished implementing variational autoencoders (VAEs)! The ELBO objective is:

$$\\mathcal{L} = \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - D_{KL}(q_\\phi(z|x) \\| p(z))$$

The first term is the reconstruction loss, and the second is the KL divergence that regularizes the latent space. The reparameterization trick $z = \\mu + \\sigma \\odot \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0,I)$ makes training possible through backpropagation! 🎨 #VAE #GenerativeModels`,
      time: '10h',
      likes: 53,
      retweets: 18,
      replies: 9,
      type: 'user-post'
    }
  ]

  // PaperCard component
  const PaperCard = ({ paper }: { paper: any }) => {
    const isExpanded = expandedAbstracts.has(paper.id)
    
    return (
      <div className="paper-card">
        <div className="paper-header">
          <div className="paper-avatar">
            {paper.authorInitial}
          </div>
          <div className="paper-author">
            {paper.author}
          </div>
        </div>
        
        <div className="paper-title">
          {paper.title}
          <span dangerouslySetInnerHTML={{ __html: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
            <polyline points="15,3 21,3 21,9"/>
            <line x1="10" y1="14" x2="21" y2="3"/>
          </svg>` }} />
        </div>
        
        <div className="paper-summary">
          {paper.summary}
        </div>
        
        <div>
          <div className={`paper-abstract ${isExpanded ? 'expanded' : ''}`}>
            {paper.abstract}
          </div>
          <button 
            className="show-more-btn"
            onClick={() => toggleAbstract(paper.id)}
          >
            {isExpanded ? 'Show less' : 'Show more'}
          </button>
        </div>
        
        <div className="paper-badge">
          <div className="paper-badge-left">
            Auto-imported from {paper.id} · {paper.citations} citations
          </div>
          <div className="paper-badge-right">
            <span dangerouslySetInnerHTML={{ __html: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 2v6h-6"/>
              <path d="M3 12a9 9 0 0 1 15-6.7L21 8"/>
              <path d="M3 22v-6h6"/>
              <path d="M21 12a9 9 0 0 1-15 6.7L3 16"/>
            </svg>` }} />
          </div>
        </div>
      </div>
    )
  }

  const navItems = [
    { 
      id: 'feed', 
      label: 'Feed', 
      icon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M8 6h8"/>
        <path d="M8 12h8"/>
        <path d="M8 18h8"/>
        <path d="M4 6h.01"/>
        <path d="M4 12h.01"/>
        <path d="M4 18h.01"/>
      </svg>`
    },
    { 
      id: 'search', 
      label: 'Search', 
      icon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <path d="m21 21-4.35-4.35"/>
      </svg>`
    },
    { 
      id: 'collections', 
      label: 'Collections', 
      icon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 3h18v18H3z"/>
        <path d="M9 9h6v6H9z"/>
        <path d="M3 9h6"/>
        <path d="M15 9h6"/>
        <path d="M3 15h6"/>
        <path d="M15 15h6"/>
      </svg>`
    },
    { 
      id: 'scholars', 
      label: 'Scholars', 
      icon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
        <circle cx="9" cy="7" r="4"/>
        <path d="m22 21-2-2"/>
        <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
      </svg>`
    },
    { 
      id: 'affiliations', 
      label: 'Affiliations', 
      icon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 21h18"/>
        <path d="M5 21V7l8-4v18"/>
        <path d="M19 21V11l-6-4"/>
        <path d="M9 9h.01"/>
        <path d="M9 13h.01"/>
        <path d="M9 17h.01"/>
        <path d="M14 13h.01"/>
        <path d="M14 17h.01"/>
      </svg>`
    },
    { 
      id: 'papers', 
      label: 'Papers Only', 
      icon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14,2 14,8 20,8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10,9 9,9 8,9"/>
      </svg>`
    },
    { 
      id: 'profile', 
      label: 'Profile', 
      icon: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>`
    }
  ]

  return (
    <div style={{ fontFamily: 'Inter, Helvetica Neue, Helvetica, Arial, sans-serif' }}>
      <style>{FEED_CSS}</style>
      
      <div className="feed-container">
        {/* Left Sidebar */}
        <div className="feed-sidebar">
          <div className="sidebar-nav">
            {navItems.map(item => (
              <div
                key={item.id}
                className={`nav-item ${activeNav === item.id ? 'active' : ''}`}
                onClick={() => setActiveNav(item.id)}
              >
                <span dangerouslySetInnerHTML={{ __html: item.icon }} />
                {item.label}
              </div>
            ))}
          </div>
          
          <div className="sidebar-cta">
            <button className="cta-button">
              Post Research
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="feed-main">
          {/* Post Composer */}
          <div className="post-composer">
            <div className="composer-avatar">
              U
            </div>
            <div className="composer-content">
              <textarea
                className="composer-textarea"
                placeholder="Share your research insights… Include arXiv URLs to automatically import papers. LaTeX supported with $...$ or $$...$$"
                value={composerText}
                onChange={(e) => setComposerText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handlePostSubmit()
                  }
                }}
              />
              <div className="composer-actions">
                <button 
                  className="post-button"
                  disabled={!composerText.trim()}
                  onClick={handlePostSubmit}
                >
                  Post
                </button>
              </div>
            </div>
          </div>

          {/* Feed List */}
          <div className="feed-list" ref={postsRef}>
            {posts.map(post => {
              if (post.type === 'pure-paper') {
                return (
                  <div key={post.id} className="feed-item pure-paper">
                    <PaperCard paper={post.paper} />
                  </div>
                )
              }

              return (
                <div key={post.id} className={`feed-item ${post.type}`}>
                  <div className="post-header">
                    <div className="post-avatar">
                      {post.avatar}
                    </div>
                    <div className="post-user-info">
                      <div className="post-name">{post.name}</div>
                      <span>·</span>
                      <div className="post-time">{post.time}</div>
                    </div>
                  </div>
                  
                  <div className="post-content">
                    {post.content}
                  </div>
                  
                  {post.type === 'paper-post' && post.paper && (
                    <PaperCard paper={post.paper} />
                  )}
                  
                  <div className="post-actions">
                    <div className="post-action">
                      <span dangerouslySetInnerHTML={{ __html: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                      </svg>` }} />
                      {post.replies || 0}
                    </div>
                    <div className="post-action">
                      <span dangerouslySetInnerHTML={{ __html: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M17 1l4 4-4 4"/>
                        <path d="M3 11V9a4 4 0 0 1 4-4h14"/>
                        <path d="M7 23l-4-4 4-4"/>
                        <path d="M21 13v2a4 4 0 0 1-4 4H3"/>
                      </svg>` }} />
                      {post.retweets || 0}
                    </div>
                    <div className="post-action">
                      <span dangerouslySetInnerHTML={{ __html: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                      </svg>` }} />
                      {post.likes || 0}
                    </div>
                    <div className="post-action">
                      <span dangerouslySetInnerHTML={{ __html: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 2L11 13"/>
                        <path d="M22 2l-7 20-4-9-9-4 20-7z"/>
                      </svg>` }} />
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
} 