import React, { useState, useEffect, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import ReactDOM from 'react-dom';
import { mockUsers } from './mockData/users';
import { mockScholars } from './mockData/scholars';
import { mockAffiliations } from './mockData/affiliations';
import { mockPapers } from './mockData/papers';
import { UserProfile } from './UserProfile';

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
    repo: unknown; // Using unknown for now since this is a dummy component
    currentUser?: string; // Optional current user email
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
        id: 'papers',
        label: 'Papers',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
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
        id: 'affiliations',
        label: 'Affiliations',
        icon: (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {/* Triangle roof */}
                <polygon points="4,10 12,4 20,10" stroke="currentColor" strokeWidth="2" fill="none" />
                {/* Circle window */}
                <circle cx="12" cy="8.5" r="1" stroke="currentColor" strokeWidth="2" fill="none" />
                {/* Columns */}
                <rect x="7" y="11" width="1.5" height="5" stroke="currentColor" strokeWidth="2" fill="none" />
                <rect x="11.25" y="11" width="1.5" height="5" stroke="currentColor" strokeWidth="2" fill="none" />
                <rect x="15" y="11" width="1.5" height="5" stroke="currentColor" strokeWidth="2" fill="none" />
                {/* Base */}
                <rect x="5" y="17" width="14" height="2" stroke="currentColor" strokeWidth="2" fill="none" />
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

export function Library({ repo: _repo, currentUser }: Library2Props) {
    const location = useLocation();
    // Determine initial nav from path
    const getNavFromPath = (pathname: string) => {
        if (pathname.includes('/scholars')) return 'scholars';
        if (pathname.includes('/collections')) return 'collections';
        if (pathname.includes('/affiliations')) return 'affiliations';
        if (pathname.includes('/papers')) return 'papers';
        if (pathname.includes('/profile')) return 'profile';
        if (pathname.includes('/search')) return 'search';
        return 'feed';
    };
    const [activeNav, setActiveNav] = useState(() => getNavFromPath(location.pathname));
    const [mathJaxLoaded, setMathJaxLoaded] = useState(false)
    const postsRef = useRef<HTMLDivElement>(null)
    const [composerText, setComposerText] = useState('')
    const [expandedAbstracts, setExpandedAbstracts] = useState<Set<string>>(new Set())
    const [scholars, setScholars] = useState(mockScholars)
    const [affiliations, setAffiliations] = useState(mockAffiliations)
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedScholar, setSelectedScholar] = useState<string | null>(null)
    const [sortBy, setSortBy] = useState<'name' | 'affiliation' | 'recentActivity' | 'papers' | 'citations' | 'hIndex'>('name')
    const [affiliationsSortBy, setAffiliationsSortBy] = useState<'name' | 'location' | 'type' | 'members' | 'papers' | 'citations'>('name')
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
    const [affiliationsSortDirection, setAffiliationsSortDirection] = useState<'asc' | 'desc'>('asc')
    const navigate = useNavigate();
    const [expandedScholarId, setExpandedScholarId] = useState<string | null>(null);
    const [hoveredScholarId, setHoveredScholarId] = useState<string | null>(null);
    const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    // Add a ref for the filter input at the top of Library
    const filterInputRef = useRef<HTMLInputElement>(null);

    // Sync activeNav with URL path
    useEffect(() => {
        const nav = getNavFromPath(location.pathname);
        setActiveNav(nav);
    }, [location.pathname]);

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

    const toggleFollow = (scholarId: string) => {
        setScholars(prev => prev.map(scholar => 
            scholar.id === scholarId 
                ? { ...scholar, isFollowing: !scholar.isFollowing }
                : scholar
        ))
    }

    const filteredScholars = scholars
        .filter(scholar => {
            // Only filter when search query is 2+ characters long
            if (searchQuery.length < 2) return true;
            
            const query = searchQuery.toLowerCase();
            return scholar.name.toLowerCase().includes(query) ||
                scholar.institution.toLowerCase().includes(query) ||
                scholar.expertise.some(exp => exp.toLowerCase().includes(query));
        })
        .sort((a, b) => {
            let aValue: string | number;
            let bValue: string | number;
            
            switch (sortBy) {
                case 'name':
                    // Sort by last name
                    const aLastName = a.name.split(' ').pop()?.toLowerCase() || '';
                    const bLastName = b.name.split(' ').pop()?.toLowerCase() || '';
                    aValue = aLastName;
                    bValue = bLastName;
                    break;
                case 'affiliation':
                    aValue = a.institution.toLowerCase();
                    bValue = b.institution.toLowerCase();
                    break;
                case 'recentActivity':
                    // Convert activity strings to numbers for sorting (lower = more recent)
                    const activityOrder = { '1 day ago': 1, '2 days ago': 2, '3 days ago': 3, '4 days ago': 4, '5 days ago': 5, '1 week ago': 7 };
                    aValue = activityOrder[a.recentActivity as keyof typeof activityOrder] || 10;
                    bValue = activityOrder[b.recentActivity as keyof typeof activityOrder] || 10;
                    break;
                case 'papers':
                    aValue = a.papers.length;
                    bValue = b.papers.length;
                    break;
                case 'citations':
                    aValue = a.totalCitations;
                    bValue = b.totalCitations;
                    break;
                case 'hIndex':
                    aValue = a.hIndex;
                    bValue = b.hIndex;
                    break;
                default:
                    return 0;
            }
            
            if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });

    const filteredAffiliations = affiliations
        .filter(affiliation => {
            // Only filter when search query is 2+ characters long
            if (searchQuery.length < 2) return true;
            
            const query = searchQuery.toLowerCase();
            return affiliation.label.toLowerCase().includes(query) ||
                affiliation.name.toLowerCase().includes(query) ||
                affiliation.location.toLowerCase().includes(query) ||
                affiliation.tags.some(tag => tag.toLowerCase().includes(query));
        })
        .sort((a, b) => {
            let aValue: string | number;
            let bValue: string | number;
            
            switch (affiliationsSortBy) {
                case 'name':
                    aValue = a.label.toLowerCase();
                    bValue = b.label.toLowerCase();
                    break;
                case 'location':
                    aValue = a.location.toLowerCase();
                    bValue = b.location.toLowerCase();
                    break;
                case 'type':
                    aValue = a.type.toLowerCase();
                    bValue = b.type.toLowerCase();
                    break;
                case 'members':
                    aValue = a.memberCount;
                    bValue = b.memberCount;
                    break;
                case 'papers':
                    aValue = a.papers;
                    bValue = b.papers;
                    break;
                case 'citations':
                    aValue = a.citations;
                    bValue = b.citations;
                    break;
                default:
                    return 0;
            }
            
            if (aValue < bValue) return affiliationsSortDirection === 'asc' ? -1 : 1;
            if (aValue > bValue) return affiliationsSortDirection === 'asc' ? 1 : -1;
            return 0;
        });

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

    // Sidebar navigation handler
    const handleNavClick = (id: string) => {
        setActiveNav(id);
        switch (id) {
            case 'scholars':
                navigate('/scholars');
                break;
            case 'affiliations':
                navigate('/affiliations');
                break;
            case 'feed':
                navigate('/library');
                break;
            case 'search':
                navigate('/search');
                break;
            case 'collections':
                navigate('/collections');
                break;
            case 'papers':
                navigate('/papers');
                break;
            case 'profile':
                navigate('/profile');
                break;
            default:
                break;
        }
    }

    // ScholarCard component
    const ScholarCard = ({ scholar, expanded, onExpand, onCollapse }: { scholar: any, expanded: boolean, onExpand: (id: string) => void, onCollapse: (id: string) => void }) => {
        const cardRef = useRef<HTMLDivElement>(null);
        const containerRef = useRef<HTMLDivElement>(null);

        // Tag logic - show all expertise tags
        const tagsToShow = scholar.expertise;

        // Robust hover logic using mouse tracking
        const handleMouseEnter = () => {
            // Clear any pending collapse timeout
            if (hoverTimeoutRef.current) {
                clearTimeout(hoverTimeoutRef.current);
                hoverTimeoutRef.current = null;
            }
            
            // Set this scholar as hovered
            setHoveredScholarId(scholar.id);
            
            // Expand immediately
            onExpand(scholar.id);
        };

        const handleMouseLeave = (event: React.MouseEvent) => {
            // Check if we're moving to a child element (like the drawer)
            const relatedTarget = event.relatedTarget as HTMLElement;
            if (containerRef.current && relatedTarget && containerRef.current.contains(relatedTarget)) {
                return; // Still within the container, don't collapse
            }

            // Set a timeout to collapse, but only if we're not hovering over the drawer
            hoverTimeoutRef.current = setTimeout(() => {
                setHoveredScholarId(null);
                onCollapse(scholar.id);
            }, 150); // Slightly longer timeout for better UX
        };

        // Check if mouse is still over the container (for drawer interactions)
        const handleContainerMouseMove = () => {
            if (hoveredScholarId === scholar.id) {
                // Mouse is still over this scholar, ensure it stays expanded
                onExpand(scholar.id);
            }
        };

        // Cleanup timeout on unmount
        useEffect(() => {
            return () => {
                if (hoverTimeoutRef.current) {
                    clearTimeout(hoverTimeoutRef.current);
                }
            };
        }, []);

        return (
            <div
                ref={containerRef}
                className="relative break-inside-avoid mb-6"
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                onMouseMove={handleContainerMouseMove}
            >
                {/* Scholar card with simplified, robust layout */}
                <div 
                    ref={cardRef}
                    className="relative w-full bg-white rounded-lg border border-gray-200 p-3 hover:shadow-md transition-all cursor-pointer"
                    style={{ minHeight: '160px' }}
                    onClick={() => navigate(`/scholars/${scholar.id}`)}
                >
                    {/* Avatar and follow button section */}
                    <div className="flex items-start gap-3 min-w-0 mb-3">
                        <div className="flex flex-col items-center flex-shrink-0" style={{ width: 60 }}>
                            <div className={`w-12 h-12 rounded-full flex items-center justify-center text-sm font-semibold ${scholar.claimed ? 'bg-primary-500 text-white' : 'bg-gray-300 text-gray-600'}`}> 
                                {scholar.initials || (scholar.name.split(' ').map((n: string) => n[0]).join('').toUpperCase())}
                            </div>
                            {scholar.claimed && (
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        toggleFollow(scholar.id);
                                    }}
                                    className={`mt-1 px-2 py-0.5 rounded-full text-[9px] uppercase tracking-wider font-semibold transition-colors ${
                                        scholar.isFollowing
                                            ? 'bg-orange-100 text-orange-700'
                                            : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                    }`}
                                >
                                    {scholar.isFollowing ? 'FOLLOWING' : 'FOLLOW'}
                                </button>
                            )}
                        </div>
                        
                        {/* Name and institution section */}
                        <div className="flex-1 min-w-0">
                            <h3 className="text-base font-semibold text-gray-900 break-words leading-tight mb-1">
                                {highlightMatchingText(scholar.name, searchQuery)}
                            </h3>
                            <p className="text-gray-600 text-sm break-words leading-tight">
                                {highlightMatchingText(scholar.institution, searchQuery)}
                            </p>
                        </div>
                    </div>
                    
                    {/* Expertise tags as flowing text with consistent spacing */}
                    <div className="w-full">
                        <div className="text-xs font-semibold text-gray-700 leading-tight break-words">
                            {tagsToShow.map((exp: string, index: number) => (
                                <React.Fragment key={exp}>
                                    {index > 0 && <span className="text-gray-400"> • </span>}
                                    <button
                                        type="button"
                                        className="hover:text-primary-600 hover:underline transition-colors cursor-pointer p-0 m-0 bg-transparent border-none font-semibold break-words"
                                        style={{ display: 'inline', background: 'none' }}
                                        onClick={e => {
                                            e.stopPropagation();
                                            setSearchQuery(exp);
                                            filterInputRef.current?.focus();
                                        }}
                                    >
                                        {highlightMatchingText(exp, searchQuery)}
                                    </button>
                                </React.Fragment>
                            ))}
                        </div>
                    </div>
                </div>
                {/* Hover drawer, absolutely positioned outside the card, never affects card layout */}
                {expanded && (
                    <div
                        className="absolute left-0 top-full -mt-4.75 w-full bg-gray-200 border-l border-r border-b border-gray-300 shadow-lg p-4 rounded-b-lg rounded-t-none"
                        style={{ 
                            zIndex: 30, 
                            pointerEvents: 'auto',
                            transform: 'translateY(-4px)' // Slight overlap for seamless look
                        }}
                    >
                        <div className="flex items-center gap-4 text-xs text-gray-600 mb-3">
                            <div>
                                <span className="font-semibold text-gray-900">{scholar.hIndex}</span>
                                <span className="ml-1">h-index</span>
                            </div>
                            <div>
                                <span className="font-semibold text-gray-900">{scholar.totalCitations.toLocaleString()}</span>
                                <span className="ml-1">citations</span>
                            </div>
                            <div>
                                <span className="font-semibold text-gray-900">{scholar.papers.length}</span>
                                <span className="ml-1">papers</span>
                            </div>
                        </div>
                        <div className="border-t border-gray-300 pt-2">
                            <h4 className="font-semibold text-gray-900 mb-2 text-sm">Recent Papers</h4>
                            <div className="space-y-1">
                                {scholar.papers.slice(0, 2).map((paper: any) => (
                                    <div key={paper.id} className="min-w-0">
                                        <a
                                            href={paper.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-xs font-medium text-gray-900 break-words leading-tight hover:text-primary-600 block"
                                            onClick={(e) => e.stopPropagation()}
                                        >
                                            {paper.title}
                                        </a>
                                        <p className="text-xs text-gray-500">{paper.year} • {paper.citations} citations</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className="mt-2 text-xs text-gray-500">
                            Active {scholar.recentActivity}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    // AffiliationsCard component


    const AffiliationsCard = ({ affiliation, isAdmin }: { affiliation: any, isAdmin: boolean }) => (
        <div 
            className="w-full min-w-[20rem] max-w-full bg-white rounded-lg border border-gray-200 p-4 hover:shadow transition-shadow overflow-hidden flex flex-col cursor-pointer"
            onClick={() => navigate(`/affiliations/${affiliation.id}`)}
        >
            <div className="flex items-start justify-between mb-2 min-w-0">
                <div className="flex items-center gap-3 min-w-0">
                    {affiliation.logo ? (
                        <img src={affiliation.logo} alt={affiliation.label} className="w-12 h-12 rounded-full object-cover flex-shrink-0" />
                    ) : (
                        <div className="w-12 h-12 bg-primary-500 text-white rounded-full flex items-center justify-center text-lg font-semibold flex-shrink-0">
                            {affiliation.initials}
                        </div>
                    )}
                    <div className="min-w-0">
                        <h3 className="text-lg font-semibold text-gray-900 break-words leading-tight">{affiliation.label}</h3>
                        <p className="text-gray-600 text-sm break-words leading-tight">{affiliation.name}</p>
                        <p className="text-gray-500 text-xs break-words leading-tight">{affiliation.location}</p>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-2 min-w-0">
                    <span className="px-3 py-0.5 rounded-full text-xs font-semibold mb-1 bg-gray-100 text-gray-600 border border-gray-200" title={affiliation.type}>{affiliation.type}</span>
                </div>
            </div>
            <div className="flex flex-wrap gap-1 mb-2">
                {affiliation.tags.map((tag: string, idx: number) => (
                    <span key={idx} className="px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded-full">{tag}</span>
                ))}
            </div>
            <div className="flex items-center gap-4 text-xs text-gray-600 mb-2">
                <div>
                    <span className="font-semibold text-gray-900">{affiliation.memberCount}</span>
                    <span className="ml-1">members</span>
                </div>
                <div>
                    <span className="font-semibold text-gray-900">{affiliation.papers}</span>
                    <span className="ml-1">papers</span>
                </div>
                <div>
                    <span className="font-semibold text-gray-900">{affiliation.citations.toLocaleString()}</span>
                    <span className="ml-1">citations</span>
                </div>
            </div>
            <div className="border-t border-gray-200 pt-2 mt-2 flex items-center justify-between">
                <span className="text-xs text-gray-500">Active {affiliation.lastActive}</span>
                <div className="flex gap-2 items-center">
                    <a 
                        href={affiliation.website} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-xs text-primary-500 hover:text-primary-600 underline"
                        onClick={(e) => e.stopPropagation()}
                    >
                        Website
                    </a>
                    {isAdmin && (
                        <>
                            <button 
                                className="px-2 py-0.5 rounded bg-gray-100 text-gray-700 text-xs border border-gray-200 hover:bg-gray-200"
                                onClick={(e) => e.stopPropagation()}
                            >
                                Merge
                            </button>
                            <button 
                                className="px-2 py-0.5 rounded bg-gray-100 text-gray-700 text-xs border border-gray-200 hover:bg-gray-200"
                                onClick={(e) => e.stopPropagation()}
                            >
                                Mark Duplicate
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    )

    // UserHoverCard component
    const UserHoverCard = ({ user, position }: { user: any, position: { x: number, y: number } }) =>
        ReactDOM.createPortal(
            <div
                className="fixed z-50 bg-white border border-gray-200 rounded-lg shadow-lg p-4 min-w-[180px] max-w-xs pointer-events-auto"
                style={{ left: position.x, top: position.y }}
            >
                <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 bg-primary-500 text-white rounded-full flex items-center justify-center text-lg font-semibold">
                        {user.avatar}
                    </div>
                    <div>
                        <div className="font-semibold text-gray-900 text-base leading-tight">{user.name}</div>
                        {user.email && <div className="text-xs text-gray-500">{user.email}</div>}
                    </div>
                </div>
                <a
                    href={`/scholars/${user.id}`}
                    className="inline-block mt-2 text-xs text-primary-600 hover:underline font-medium"
                >
                    View profile
                </a>
            </div>,
            document.body
        );

    // PapersTable component
    const [papersSort, setPapersSort] = useState<{col: string, dir: 'asc'|'desc'}>({col: 'title', dir: 'asc'});
    const [papers, setPapers] = useState(mockPapers);
    const handleSort = (col: string) => {
        setPapersSort(prev => {
            const dir = prev.col === col && prev.dir === 'asc' ? 'desc' : 'asc';
            return { col, dir };
        });
        setPapers(prev => {
            const newDir = papersSort.col === col && papersSort.dir === 'asc' ? 'desc' : 'asc';
            const sorted = [...prev].sort((a, b) => {
                if (col === 'title') {
                    return (a.title.localeCompare(b.title)) * (newDir === 'asc' ? 1 : -1);
                }
                if (col === 'stars') {
                    return (a.stars - b.stars) * (newDir === 'asc' ? 1 : -1);
                }
                return 0;
            });
            return sorted;
        });
    };
    const toggleStar = (id: string) => {
        setPapers(prev => prev.map(p => p.id === id ? { ...p, starred: !p.starred, stars: p.starred ? p.stars - 1 : p.stars + 1 } : p));
    };
    const [hoveredUser, setHoveredUser] = useState<{ user: any, position: { x: number, y: number } } | null>(null);
    
    // Helper function to highlight matching text in tags
    const highlightMatchingText = (text: string, query: string) => {
        if (query.length < 2) return text;
        
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        const parts = text.split(regex);
        
        return parts.map((part, index) => 
            regex.test(part) ? (
                <strong key={index} className="font-bold">{part}</strong>
            ) : (
                part
            )
        );
    };
    
    const PapersTable = () => (
        <div className="overflow-x-auto relative">
            {hoveredUser && <UserHoverCard user={hoveredUser.user} position={hoveredUser.position} />}
            <table className="min-w-full bg-white border border-gray-200 rounded-lg text-sm">
                <thead>
                    <tr className="bg-gray-50">
                        <th className="px-4 py-2 text-left cursor-pointer sticky left-0 z-10 bg-white border-r border-gray-200" onClick={() => handleSort('title')}>
                            Title
                            <span className="ml-1 align-middle">{papersSort.col === 'title' ? (papersSort.dir === 'asc' ? '▲' : '▼') : ''}</span>
                        </th>
                        <th className="px-4 py-2 text-left">Authors</th>
                        <th className="px-4 py-2 text-left">Affiliations</th>
                        <th className="px-4 py-2 text-left">Tags</th>
                        <th className="px-4 py-2 text-left">Read by</th>
                        <th className="px-4 py-2 text-left">Link</th>
                        <th className="px-4 py-2 text-left">Queued</th>
                        <th className="px-4 py-2 text-left cursor-pointer" onClick={() => handleSort('stars')}>
                            Stars
                            <span className="ml-1 align-middle">{papersSort.col === 'stars' ? (papersSort.dir === 'asc' ? '▲' : '▼') : ''}</span>
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {papers.map(paper => (
                        <tr key={paper.id} className="border-t border-gray-100 hover:bg-gray-50">
                            <td className="px-4 py-2 whitespace-nowrap flex items-center gap-2 sticky left-0 z-10 bg-white border-r border-gray-200">
                                <button onClick={() => toggleStar(paper.id)} className="focus:outline-none">
                                    {paper.starred ? (
                                        <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.967a1 1 0 00.95.69h4.175c.969 0 1.371 1.24.588 1.81l-3.38 2.455a1 1 0 00-.364 1.118l1.287 3.966c.3.922-.755 1.688-1.54 1.118l-3.38-2.454a1 1 0 00-1.175 0l-3.38 2.454c-.784.57-1.838-.196-1.54-1.118l1.287-3.966a1 1 0 00-.364-1.118L2.05 9.394c-.783-.57-.38-1.81.588-1.81h4.175a1 1 0 00.95-.69l1.286-3.967z"/></svg>
                                    ) : (
                                        <svg className="w-4 h-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 20 20"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 15l-5.878 3.09 1.122-6.545L.488 6.91l6.564-.955L10 0l2.948 5.955 6.564.955-4.756 4.635 1.122 6.545z"/></svg>
                                    )}
                                </button>
                                <span>{paper.title}</span>
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap">
                                {paper.authors.map((authorId: string, idx: number) => {
                                    const author = mockScholars.find(s => s.id === authorId);
                                    return author ? (
                                        <a key={author.id} href={`/scholars/${author.id}`} className="text-primary-600 hover:underline mr-1">
                                            {author.name}{idx < paper.authors.length - 1 ? ',' : ''}
                                        </a>
                                    ) : null;
                                })}
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap">
                                {paper.affiliations.map((affId: string, idx: number) => {
                                    const aff = mockAffiliations.find(a => a.id === affId);
                                    return aff ? (
                                        <a key={aff.id} href={`/affiliations/${aff.id}`} className="text-primary-600 hover:underline mr-1">
                                            {aff.label}{idx < paper.affiliations.length - 1 ? ',' : ''}
                                        </a>
                                    ) : null;
                                })}
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap">
                                {paper.tags.map((tag, idx) => (
                                    <span key={idx} className="inline-block bg-gray-100 text-gray-700 text-xs rounded-full px-2 py-0.5 mr-1 mb-0.5">{tag}</span>
                                ))}
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap">
                                <div className="flex -space-x-2">
                                    {Array.isArray(paper.readBy) && paper.readBy.length > 0 ? paper.readBy.map((user: any) => (
                                        <span
                                            key={user.id}
                                            className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-primary-200 text-primary-800 text-xs font-bold border-2 border-white cursor-pointer"
                                            title={user.name}
                                            onMouseEnter={e => {
                                                const rect = (e.target as HTMLElement).getBoundingClientRect();
                                                setHoveredUser({
                                                    user,
                                                    position: {
                                                        x: rect.right + 8, // right edge of circle + 8px
                                                        y: rect.top // top edge of circle (viewport coordinates)
                                                    }
                                                });
                                            }}
                                            onMouseLeave={() => setHoveredUser(null)}
                                        >
                                            {user.avatar}
                                        </span>
                                    )) : null}
                                </div>
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap">
                                <a href={paper.link} target="_blank" rel="noopener noreferrer" className="text-primary-500 hover:text-primary-600">
                                    <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 3h7v7m0 0L10 21l-7-7 11-11z"/></svg>
                                </a>
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap">
                                <div className="flex -space-x-2">
                                    {Array.isArray(paper.queued) && paper.queued.length > 0 ? paper.queued.map((user: any) => (
                                        <span
                                            key={user.id}
                                            className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-xs font-bold border-2 border-white cursor-pointer"
                                            title={user.name}
                                            onMouseEnter={e => {
                                                const rect = (e.target as HTMLElement).getBoundingClientRect();
                                                setHoveredUser({
                                                    user,
                                                    position: {
                                                        x: rect.right + 8,
                                                        y: rect.top
                                                    }
                                                });
                                            }}
                                            onMouseLeave={() => setHoveredUser(null)}
                                        >
                                            {user.avatar}
                                        </span>
                                    )) : null}
                                </div>
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap text-center">{paper.stars}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );

    // Render different views based on activeNav
    const renderContent = () => {
        switch (activeNav) {
            case 'scholars':
                return (
                    <div className="p-4">
                        {/* Filter and Sort Controls: left-aligned, max width */}
                        <div className="max-w-4xl mb-4 space-y-4">
                            {/* Search Bar */}
                            <div className="relative">
                                <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <circle cx="11" cy="11" r="8" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m21 21-4.35-4.35" />
                                </svg>
                                <input
                                    ref={filterInputRef}
                                    type="text"
                                    placeholder="Filter by name, institution, or expertise..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                />
                            </div>
                            {/* Sort Controls */}
                            <div className="flex items-center gap-4 text-sm">
                                <span className="text-gray-600 font-medium">Sort by:</span>
                                <div className="flex gap-2">
                                    {[
                                        { key: 'name', label: 'Name' },
                                        { key: 'affiliation', label: 'Affiliation' },
                                        { key: 'recentActivity', label: 'Recent Activity' },
                                        { key: 'papers', label: 'Papers' },
                                        { key: 'citations', label: 'Citations' },
                                        { key: 'hIndex', label: 'H-index' }
                                    ].map(({ key, label }) => (
                                        <button
                                            key={key}
                                            onClick={() => {
                                                if (sortBy === key) {
                                                    setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
                                                } else {
                                                    setSortBy(key as any);
                                                    setSortDirection('asc');
                                                }
                                            }}
                                            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                                                sortBy === key
                                                    ? 'bg-primary-100 text-primary-700 border border-primary-200'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
                                            }`}
                                        >
                                            {label}
                                            {sortBy === key && (
                                                <span className="ml-1">
                                                    {sortDirection === 'asc' ? '↑' : '↓'}
                                                </span>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                        {/* Scholars Grid: left-to-right flow like text */}
                        <div className="w-full px-2">
                            <div
                                className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
                            >
                                {filteredScholars.map(scholar => (
                                    <ScholarCard
                                        key={scholar.id}
                                        scholar={scholar}
                                        expanded={expandedScholarId === scholar.id}
                                        onExpand={id => setExpandedScholarId(id)}
                                        onCollapse={id => {
                                            if (expandedScholarId === id) setExpandedScholarId(null);
                                        }}
                                    />
                                ))}
                            </div>
                            {filteredScholars.length === 0 && (
                                <div className="text-center py-8">
                                    <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                    </svg>
                                    <p className="text-gray-500">No scholars found matching your filter.</p>
                                </div>
                            )}
                        </div>
                    </div>
                )

            case 'affiliations':
                return (
                    <div className="p-4">
                        {/* Filter and Sort Controls: left-aligned, max width */}
                        <div className="max-w-4xl mb-4 space-y-4">
                            {/* Search Bar */}
                            <div className="relative">
                                <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <circle cx="11" cy="11" r="8" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="m21 21-4.35-4.35" />
                                </svg>
                                <input
                                    ref={filterInputRef}
                                    type="text"
                                    placeholder="Filter by name, location, or tags..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                />
                            </div>
                            {/* Sort Controls */}
                            <div className="flex items-center gap-4 text-sm">
                                <span className="text-gray-600 font-medium">Sort by:</span>
                                <div className="flex gap-2">
                                    {[
                                        { key: 'name', label: 'Name' },
                                        { key: 'location', label: 'Location' },
                                        { key: 'type', label: 'Type' },
                                        { key: 'members', label: 'Members' },
                                        { key: 'papers', label: 'Papers' },
                                        { key: 'citations', label: 'Citations' }
                                    ].map(({ key, label }) => (
                                        <button
                                            key={key}
                                            onClick={() => {
                                                if (affiliationsSortBy === key) {
                                                    setAffiliationsSortDirection(affiliationsSortDirection === 'asc' ? 'desc' : 'asc');
                                                } else {
                                                    setAffiliationsSortBy(key as any);
                                                    setAffiliationsSortDirection('asc');
                                                }
                                            }}
                                            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                                                affiliationsSortBy === key
                                                    ? 'bg-primary-100 text-primary-700 border border-primary-200'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
                                            }`}
                                        >
                                            {label}
                                            {affiliationsSortBy === key && (
                                                <span className="ml-1">
                                                    {affiliationsSortDirection === 'asc' ? '↑' : '↓'}
                                                </span>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                        {/* Affiliations Grid: left-to-right flow like text */}
                        <div className="w-full px-2">
                            <div
                                className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
                            >
                                {filteredAffiliations.map(affiliation => (
                                    <AffiliationsCard key={affiliation.id} affiliation={affiliation} isAdmin={affiliation.isAdmin} />
                                ))}
                            </div>
                            {filteredAffiliations.length === 0 && (
                                <div className="text-center py-8">
                                    <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                    </svg>
                                    <p className="text-gray-500">No affiliations found matching your filter.</p>
                                </div>
                            )}
                        </div>
                    </div>
                )

            case 'papers':
                return (
                    <div className="p-6">
                        <div className="max-w-6xl mx-auto">
                            <div className="mb-6">
                                <h1 className="text-2xl font-bold text-gray-900 mb-2">Papers</h1>
                                <p className="text-gray-600">Browse and sort all papers in the system</p>
                            </div>
                            <PapersTable />
                        </div>
                    </div>
                )

            case 'profile':
                return (
                    <div className="p-6">
                        <div className="max-w-6xl mx-auto">
                            <UserProfile repo={_repo} currentUser={currentUser || "alice@example.com"} />
                        </div>
                    </div>
                )

            case 'feed':
            default:
                return (
                    <>
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
                    </>
                )
        }
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
                                    onClick={() => handleNavClick(item.id)}
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
                    {renderContent()}
                </div>
            </div>
        </div>
    )
}
