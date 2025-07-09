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

// CSS for feed page
const FEED_CSS = `
.feed-container {
  display: flex;
  max-width: 1200px;
  margin: 0 auto;
  min-height: calc(100vh - 60px);
}

.feed-sidebar {
  width: 250px;
  background: #fff;
  border-right: 1px solid #e1e8ed;
  padding: 20px 0;
  position: sticky;
  top: 0;
  height: fit-content;
}

.feed-main {
  flex: 1;
  background: #fff;
}

.feed-header {
  padding: 15px 20px;
  border-bottom: 1px solid #e1e8ed;
  background: #fff;
  position: sticky;
  top: 0;
  z-index: 10;
}

.feed-header h1 {
  font-size: 20px;
  font-weight: 700;
  color: #14171a;
  margin: 0;
}

.nav-item {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: #14171a;
  text-decoration: none;
  font-weight: 500;
  font-size: 15px;
  transition: background-color 0.2s ease;
  border-radius: 25px;
  margin: 0 10px 4px 10px;
}

.nav-item:hover {
  background-color: #f7f9fa;
}

.nav-item.active {
  font-weight: 700;
}

.nav-item svg {
  margin-right: 12px;
  width: 20px;
  height: 20px;
}

.post {
  padding: 15px 20px;
  border-bottom: 1px solid #e1e8ed;
  transition: background-color 0.2s ease;
}

.post:hover {
  background-color: #f7f9fa;
}

.post-header {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.post-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #1da1f2;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 700;
  font-size: 18px;
  margin-right: 12px;
}

.post-user-info {
  flex: 1;
}

.post-name {
  font-weight: 700;
  color: #14171a;
  font-size: 15px;
  margin: 0;
}

.post-username {
  color: #657786;
  font-size: 14px;
  margin: 0;
}

.post-time {
  color: #657786;
  font-size: 14px;
}

.post-content {
  color: #14171a;
  font-size: 15px;
  line-height: 1.5;
  margin-bottom: 12px;
}

.post-content .math {
  display: inline-block;
  margin: 0 2px;
}

.post-content .math-display {
  display: block;
  text-align: center;
  margin: 10px 0;
}

.post-actions {
  display: flex;
  gap: 60px;
  color: #657786;
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
  color: #1da1f2;
}

.post-action svg {
  width: 16px;
  height: 16px;
}

.compose-button {
  background: #1da1f2;
  color: white;
  border: none;
  border-radius: 25px;
  padding: 15px 20px;
  font-weight: 700;
  font-size: 15px;
  cursor: pointer;
  width: calc(100% - 20px);
  margin: 0 10px 20px 10px;
  transition: background-color 0.2s ease;
}

.compose-button:hover {
  background: #1a91da;
}

.trending-section {
  padding: 15px 20px;
  border-bottom: 1px solid #e1e8ed;
}

.trending-title {
  font-weight: 700;
  color: #14171a;
  font-size: 15px;
  margin-bottom: 12px;
}

.trending-item {
  padding: 8px 0;
  cursor: pointer;
}

.trending-topic {
  font-weight: 700;
  color: #14171a;
  font-size: 14px;
  margin-bottom: 2px;
}

.trending-count {
  color: #657786;
  font-size: 12px;
}

.paper-preview {
  background: #f8f9fa;
  border: 1px solid #e1e8ed;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
}

.paper-title {
  font-weight: 600;
  color: #14171a;
  font-size: 14px;
  margin-bottom: 4px;
}

.paper-authors {
  color: #657786;
  font-size: 12px;
  margin-bottom: 4px;
}

.paper-abstract {
  color: #14171a;
  font-size: 13px;
  line-height: 1.4;
}
`

interface LibraryProps {
  repo: unknown // Using unknown for now since this is a dummy component
}

export function Library({ repo: _repo }: LibraryProps) {
  const [activeNav, setActiveNav] = useState('home')
  const [mathJaxLoaded, setMathJaxLoaded] = useState(false)
  const postsRef = useRef<HTMLDivElement>(null)

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
      replies: 3
    },
    {
      id: 2,
      name: 'Prof. Bob Chen',
      username: '@bobchen',
      avatar: 'B',
      content: `Excited to share our new paper on reinforcement learning with continuous action spaces! We introduce a novel policy gradient method that uses the natural gradient $\\nabla_\\theta J(\\theta) = F^{-1}(\\theta) \\nabla_\\theta J(\\theta)$ where $F(\\theta)$ is the Fisher information matrix. This leads to more stable training and better sample efficiency. The key equation is:

$$\\theta_{t+1} = \\theta_t + \\alpha F^{-1}(\\theta_t) \\nabla_\\theta J(\\theta_t)$$

Our experiments show 40% improvement in sample efficiency on continuous control tasks! 📊 #RL #PolicyGradients`,
      time: '4h',
      likes: 42,
      retweets: 15,
      replies: 7
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
      replies: 12
    },
    {
      id: 4,
      name: 'David Kim',
      username: '@davidkim',
      avatar: 'D',
      content: `Working on transformer architectures for time series forecasting. The key challenge is handling the quadratic complexity $O(n^2)$ of self-attention. We're exploring sparse attention patterns where each position only attends to $O(\\log n)$ other positions. The attention complexity becomes $O(n \\log n)$, making it much more efficient for long sequences! ⚡ #Transformers #TimeSeries`,
      time: '8h',
      likes: 31,
      retweets: 9,
      replies: 5
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
      replies: 9
    }
  ]

  const trendingTopics = [
    { topic: '#AttentionMechanisms', count: '12.5K posts' },
    { topic: '#ReinforcementLearning', count: '8.2K posts' },
    { topic: '#NeuralNetworks', count: '5.7K posts' },
    { topic: '#Transformers', count: '4.1K posts' },
    { topic: '#GenerativeModels', count: '3.8K posts' }
  ]

  const navItems = [
    { id: 'home', label: 'Home', icon: '🏠' },
    { id: 'explore', label: 'Explore', icon: '🔍' },
    { id: 'notifications', label: 'Notifications', icon: '🔔' },
    { id: 'messages', label: 'Messages', icon: '💬' },
    { id: 'bookmarks', label: 'Bookmarks', icon: '🔖' },
    { id: 'profile', label: 'Profile', icon: '👤' }
  ]

  return (
    <div style={{ fontFamily: 'Arial, sans-serif' }}>
      <style>{FEED_CSS}</style>
      
      <div className="feed-container">
        {/* Left Sidebar */}
        <div className="feed-sidebar">
          <button className="compose-button">
            Post
          </button>
          
          {navItems.map(item => (
            <div
              key={item.id}
              className={`nav-item ${activeNav === item.id ? 'active' : ''}`}
              onClick={() => setActiveNav(item.id)}
            >
              <span>{item.icon}</span>
              {item.label}
            </div>
          ))}
        </div>

        {/* Main Content */}
        <div className="feed-main">
          <div className="feed-header">
            <h1>Home</h1>
          </div>

          {/* Posts Feed */}
          <div ref={postsRef}>
            {posts.map(post => (
              <div key={post.id} className="post">
                <div className="post-header">
                  <div className="post-avatar">
                    {post.avatar}
                  </div>
                  <div className="post-user-info">
                    <div className="post-name">{post.name}</div>
                    <div className="post-username">{post.username}</div>
                  </div>
                  <div className="post-time">{post.time}</div>
                </div>
                
                <div className="post-content">
                  {post.content}
                </div>
                
                <div className="post-actions">
                  <div className="post-action">
                    <span>💬</span>
                    {post.replies}
                  </div>
                  <div className="post-action">
                    <span>🔄</span>
                    {post.retweets}
                  </div>
                  <div className="post-action">
                    <span>❤️</span>
                    {post.likes}
                  </div>
                  <div className="post-action">
                    <span>📤</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Trending Section */}
          <div className="trending-section">
            <div className="trending-title">Trending</div>
            {trendingTopics.map((topic, index) => (
              <div key={index} className="trending-item">
                <div className="trending-topic">{topic.topic}</div>
                <div className="trending-count">{topic.count}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
} 