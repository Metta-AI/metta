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

interface Library2Props {
    repo: unknown // Using unknown for now since this is a dummy component
}

// Navigation items with simplified gray outline icons
const navItems = [
    {
        id: 'feed',
        label: 'Feed',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
        )
    },
    {
        id: 'search',
        label: 'Search',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="8" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m21 21-4.35-4.35" />
            </svg>
        )
    },
    {
        id: 'collections',
        label: 'Collections',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
        )
    },
    {
        id: 'scholars',
        label: 'Scholars',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
        )
    },
    {
        id: 'papers',
        label: 'Papers Only',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
        )
    },
    {
        id: 'profile',
        label: 'Profile',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
        )
    }
]

export function Library({ repo: _repo }: Library2Props) {
    const [activeNav, setActiveNav] = useState('feed')
    const [mathJaxLoaded, setMathJaxLoaded] = useState(false)
    const postsRef = useRef<HTMLDivElement>(null)
    const [composerText, setComposerText] = useState('')
    const [expandedAbstracts, setExpandedAbstracts] = useState<Set<string>>(new Set())

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
            const renderMath = () => {
                if (window.MathJax?.typesetPromise) {
                    window.MathJax.typesetPromise([postsRef.current!]).then(() => {
                        console.log('MathJax rendering completed successfully')
                    }).catch((err: any) => {
                        console.error('MathJax error:', err)
                    })
                }
            }

            // Try immediately and after delays to ensure DOM is ready
            renderMath()
            setTimeout(renderMath, 100)
            setTimeout(renderMath, 500)
        }
    }, [mathJaxLoaded])

    const handlePostSubmit = () => {
        if (composerText.trim()) {
            console.log('Posting:', composerText)
            setComposerText('')
        }
    }

    const toggleAbstract = (paperId: string) => {
        const newExpanded = new Set(expandedAbstracts)
        if (newExpanded.has(paperId)) {
            newExpanded.delete(paperId)
        } else {
            newExpanded.add(paperId)
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
            <div className="bg-white rounded-lg border border-gray-200 p-4 mt-3">
                <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-primary-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                        {paper.authorInitial}
                    </div>
                    <div className="text-sm text-gray-600">
                        {paper.author}
                    </div>
                </div>

                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {paper.title}
                </h3>

                <p className="text-sm text-gray-600 mb-3">
                    {paper.summary}
                </p>

                <div className="flex items-center gap-4 text-xs text-gray-500 mb-3">
                    <span>{paper.citations.toLocaleString()} citations</span>
                    <a
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary-500 hover:text-primary-600 underline"
                    >
                        View Paper
                    </a>
                </div>

                <button
                    onClick={() => toggleAbstract(paper.id)}
                    className="text-sm text-primary-500 hover:text-primary-600 font-medium"
                >
                    {isExpanded ? 'Hide Abstract' : 'Show Abstract'}
                </button>

                {isExpanded && (
                    <div className="mt-3 p-3 bg-gray-50 rounded-md text-sm text-gray-700 leading-relaxed">
                        {paper.abstract}
                    </div>
                )}
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gray-50 font-inter">
            <div className="flex min-h-screen">
                {/* Left Sidebar */}
                <div className="w-64 bg-white border-r border-gray-200 fixed top-0 left-0 h-full flex flex-col z-10">
                    <div className="flex-1 py-4">
                        <div className="px-4 mb-6 flex items-center gap-3">
                            {/* Bookshelf Icon (from /library) */}
                            <svg className="w-7 h-7 text-gray-400" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M3 18h18" />
                                <rect x="3" y="8" width="3.6" height="10" />
                                <rect x="6.6" y="6" width="3.6" height="12" />
                                <rect x="10.2" y="9" width="3.6" height="9" />
                                <rect x="13.8" y="7" width="3.6" height="11" />
                                <rect x="17.4" y="5" width="3.6" height="13" />
                            </svg>
                            <h1 className="text-xl font-bold text-gray-900">Library</h1>
                        </div>

                        <nav className="space-y-1">
                            {navItems.map(item => (
                                <button
                                    key={item.id}
                                    onClick={() => setActiveNav(item.id)}
                                    className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${activeNav === item.id
                                        ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-500'
                                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                                        }`}
                                >
                                    {item.icon}
                                    <span className="font-medium">{item.label}</span>
                                </button>
                            ))}
                        </nav>
                    </div>

                    <div className="p-4 border-t border-gray-200">
                        <button className="w-full bg-primary-500 text-white py-2 px-4 rounded-lg font-medium hover:bg-primary-600 transition-colors">
                            Post Research
                        </button>
                    </div>
                </div>

                {/* Main Content */}
                <div className="flex-1 ml-64">
                    {/* Post Composer */}
                    <div className="bg-white border-b border-gray-200 p-6">
                        <div className="flex gap-4">
                            <div className="w-10 h-10 bg-primary-500 text-white rounded-full flex items-center justify-center font-semibold text-sm flex-shrink-0">
                                U
                            </div>
                            <div className="flex-1">
                                <textarea
                                    className="w-full min-h-[80px] max-h-32 border-0 resize-none focus:ring-0 focus:outline-none text-gray-900 placeholder-gray-500"
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
                                <div className="flex justify-end mt-3">
                                    <button
                                        className={`px-6 py-2 rounded-full font-medium transition-colors ${composerText.trim()
                                            ? 'bg-primary-500 text-white hover:bg-primary-600'
                                            : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                                            }`}
                                        disabled={!composerText.trim()}
                                        onClick={handlePostSubmit}
                                    >
                                        Post
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Feed List */}
                    <div className="max-w-2xl mx-auto" ref={postsRef}>
                        {posts.map(post => {
                            if (post.type === 'pure-paper') {
                                return (
                                    <div key={post.id} className="bg-white border-b border-gray-200 p-6">
                                        <PaperCard paper={post.paper} />
                                    </div>
                                )
                            }

                            return (
                                <div key={post.id} className={`border-b border-gray-200 p-6 ${post.type === 'user-post' ? 'bg-blue-50' :
                                    post.type === 'paper-post' ? 'bg-orange-50' : 'bg-white'
                                    }`}>
                                    <div className="flex items-center gap-3 mb-4">
                                        <div className="w-8 h-8 bg-primary-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                                            {post.avatar}
                                        </div>
                                        <div className="flex items-center gap-2 text-sm">
                                            <span className="font-semibold text-gray-900">{post.name}</span>
                                            <span className="text-gray-500">·</span>
                                            <span className="text-gray-500">{post.time}</span>
                                        </div>
                                    </div>

                                    <div className="text-gray-900 leading-relaxed mb-4 whitespace-pre-wrap">
                                        {post.content}
                                    </div>

                                    {post.type === 'paper-post' && post.paper && (
                                        <PaperCard paper={post.paper} />
                                    )}

                                    <div className="flex items-center gap-6 text-gray-500">
                                        <button className="flex items-center gap-2 hover:text-gray-700 transition-colors">
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                            </svg>
                                            <span className="text-sm">{post.replies || 0}</span>
                                        </button>
                                        <button className="flex items-center gap-2 hover:text-gray-700 transition-colors">
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                                            </svg>
                                            <span className="text-sm">{post.retweets || 0}</span>
                                        </button>
                                        <button className="flex items-center gap-2 hover:text-red-500 transition-colors">
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                                            </svg>
                                            <span className="text-sm">{post.likes || 0}</span>
                                        </button>
                                        <button className="flex items-center gap-2 hover:text-gray-700 transition-colors">
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
                                            </svg>
                                        </button>
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
