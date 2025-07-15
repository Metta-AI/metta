import { useState, useEffect, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { mockScholars } from './mockData/scholars';
import { mockAffiliations } from './mockData/affiliations';
import { mockPapers } from './mockData/papers';
import { AuthorProfile } from './AuthorProfile';
import { AffiliationProfile } from './AffiliationProfile';
import { AuthorsView, AffiliationsView, PapersView, FeedView, ProfileView } from './components/library/views';

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
        id: 'authors',
        label: 'Authors',
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
        if (pathname.includes('/authors')) return 'authors';
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

    const [authors, setAuthors] = useState(mockScholars)
    const [affiliations] = useState(mockAffiliations)
    const [searchQuery, setSearchQuery] = useState('')
    const [papersSearchQuery, setPapersSearchQuery] = useState('')
    const [sortBy, setSortBy] = useState<'name' | 'affiliation' | 'recentActivity' | 'papers' | 'citations' | 'hIndex'>('name')
    const [affiliationsSortBy, setAffiliationsSortBy] = useState<'name' | 'location' | 'type' | 'members' | 'papers' | 'citations'>('name')
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
    const [affiliationsSortDirection, setAffiliationsSortDirection] = useState<'asc' | 'desc'>('asc')
    const navigate = useNavigate();
    const [expandedAuthorId, setExpandedAuthorId] = useState<string | null>(null);
    // Add refs for the filter inputs
    const filterInputRef = useRef<HTMLInputElement>(null);
    const papersFilterInputRef = useRef<HTMLInputElement>(null);
    // Add state for overlay modals
    const [overlayAuthorId, setOverlayAuthorId] = useState<string | null>(null);
    const [overlayAffiliationId, setOverlayAffiliationId] = useState<string | null>(null);

    // Overlay close handler
    const closeOverlay = () => {
        setOverlayAuthorId(null);
        setOverlayAffiliationId(null);
    };

    // ESC key closes overlay
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') closeOverlay();
        };
        
        // Only add listener when overlay is active
        if (overlayAuthorId || overlayAffiliationId) {
            window.addEventListener('keydown', handleKeyDown);
            return () => window.removeEventListener('keydown', handleKeyDown);
        }
    }, [overlayAuthorId, overlayAffiliationId]);

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



        const toggleFollow = (authorId: string) => {
        setAuthors(prev => prev.map(author =>
            author.id === authorId
                ? { ...author, isFollowing: !author.isFollowing }
                : author
        ))
    }

    const filteredAuthors = authors
        .filter(author => {
            // Only filter when search query is 2+ characters long
            if (searchQuery.length < 2) return true;
            
            const query = searchQuery.toLowerCase();
            return author.name.toLowerCase().includes(query) ||
                author.institution.toLowerCase().includes(query) ||
                author.expertise.some(exp => exp.toLowerCase().includes(query));
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





    // Sidebar navigation handler
    const handleNavClick = (id: string) => {
        setActiveNav(id);
        switch (id) {
                    case 'authors':
            navigate('/authors');
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







    const toggleStar = (id: string) => {
        // This function is now passed to PapersView component
        // The actual state management is handled within PapersView
        console.log('Toggle star for paper:', id);
    };
    

    


    // Render different views based on activeNav
    const renderContent = () => {
        switch (activeNav) {
                        case 'authors':
                return (
                    <AuthorsView
                        filteredAuthors={filteredAuthors}
                        searchQuery={searchQuery}
                        sortBy={sortBy}
                        sortDirection={sortDirection}
                        expandedAuthorId={expandedAuthorId}
                        filterInputRef={filterInputRef}
                        onSearchChange={setSearchQuery}
                        onSortChange={(key) => setSortBy(key as any)}
                        onSortDirectionChange={setSortDirection}
                        onExpandAuthor={setExpandedAuthorId}
                        onCollapseAuthor={(id) => {
                            if (expandedAuthorId === id) setExpandedAuthorId(null);
                        }}
                        onToggleFollow={toggleFollow}
                        onTagClick={(tag) => setSearchQuery(tag)}
                        onCardClick={setOverlayAuthorId}
                    />
                )

            case 'affiliations':
                return (
                    <AffiliationsView
                        filteredAffiliations={filteredAffiliations}
                        searchQuery={searchQuery}
                        sortBy={affiliationsSortBy}
                        sortDirection={affiliationsSortDirection}
                        filterInputRef={filterInputRef}
                        onSearchChange={setSearchQuery}
                        onSortChange={(key) => setAffiliationsSortBy(key as any)}
                        onSortDirectionChange={setAffiliationsSortDirection}
                        onCardClick={setOverlayAffiliationId}
                    />
                )

            case 'papers':
                return (
                    <PapersView
                        papers={mockPapers}
                        scholars={authors}
                        affiliations={affiliations}
                        onToggleStar={toggleStar}
                        searchQuery={papersSearchQuery}
                        onSearchChange={setPapersSearchQuery}
                        filterInputRef={papersFilterInputRef}
                        onShowScholarOverlay={setOverlayAuthorId}
                        onShowAffiliationOverlay={setOverlayAffiliationId}
                    />
                )

            case 'profile':
                return (
                    <ProfileView 
                        repo={_repo} 
                        currentUser={currentUser || "alice@example.com"} 
                    />
                )

            case 'feed':
            default:
                return (
                    <FeedView
                        mathJaxLoaded={mathJaxLoaded}
                        postsRef={postsRef}
                        onPostsChange={() => {
                            // Trigger MathJax re-rendering when posts change
                            if (mathJaxLoaded && postsRef.current && window.MathJax?.typesetPromise) {
                                setTimeout(() => {
                                    const mathJax = window.MathJax;
                                    const postsElement = postsRef.current;
                                    if (mathJax?.typesetPromise && postsElement) {
                                        mathJax.typesetPromise([postsElement]).then(() => {
                                            console.log('MathJax re-rendered after post change');
                                        }).catch(err => {
                                            console.error('MathJax re-rendering error:', err);
                                        });
                                    }
                                }, 500);
                                            }
                                        }}
                                    />
                )
        }
    }

    return (
        <div className="min-h-screen bg-gray-50 font-inter">
            <div className="flex min-h-screen">
                {/* Left Sidebar */}
                <div className="w-64 bg-white border-r border-gray-200 fixed top-0 left-0 h-full flex flex-col z-10">
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

                {/* Main Content */}
                <div className="flex-1 ml-64">
                    {renderContent()}
                </div>
            </div>
            {(overlayAuthorId || overlayAffiliationId) && (
                <div
                    className="fixed inset-0 z-50 flex items-start justify-center bg-black bg-opacity-40 pt-12 pb-12"
                    onClick={closeOverlay}
                    style={{ animation: 'fadeIn 0.2s' }}
                >
                    <div
                        className="bg-white rounded-lg shadow-xl max-w-2xl max-h-[80vh] overflow-y-auto"
                        style={{ padding: '2rem 0.5rem', margin: 'auto 0' }}
                        onClick={e => e.stopPropagation()}
                        tabIndex={-1}
                    >
                        {overlayAuthorId && (
                            <AuthorProfile
                                author={authors.find(s => s.id === overlayAuthorId)}
                                onClose={closeOverlay}
                            />
                        )}
                        {overlayAffiliationId && (
                            <AffiliationProfile
                                affiliation={affiliations.find(a => a.id === overlayAffiliationId)}
                                onClose={closeOverlay}
                            />
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
