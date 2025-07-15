import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { mockScholars } from './mockData/scholars';
import { mockPapers } from './mockData/papers';
import { mockAffiliations } from './mockData/affiliations';

interface AuthorProfileProps {
    repo?: unknown;
    author?: any;
    onClose?: () => void;
}

export function AuthorProfile({ repo: _repo, author: propAuthor, onClose }: AuthorProfileProps) {
    const { authorId } = useParams<{ authorId: string }>();
    const navigate = useNavigate();
    const [author, setAuthor] = useState<any>(propAuthor || null);
    const [authorPapers, setAuthorPapers] = useState<any[]>([]);
    const [activeTab, setActiveTab] = useState<'overview' | 'papers' | 'network'>('overview');

    useEffect(() => {
        if (propAuthor) {
            setAuthor(propAuthor);
            const papers = mockPapers.filter(paper => 
                paper.authors.some(authorId => authorId === propAuthor.id)
            );
            setAuthorPapers(papers);
        } else if (authorId) {
            const foundAuthor = mockScholars.find(s => s.id === authorId);
            if (foundAuthor) {
                setAuthor(foundAuthor);
                const papers = mockPapers.filter(paper => 
                    paper.authors.some(paperAuthorId => paperAuthorId === authorId)
                );
                setAuthorPapers(papers);
            }
        }
    }, [propAuthor, authorId]);

    if (!author) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 mb-2">Author not found</h1>
                    <p className="text-gray-600 mb-4">The author you're looking for doesn't exist.</p>
                    <button 
                        onClick={() => navigate('/authors')}
                        className="bg-primary-500 text-white px-4 py-2 rounded-lg hover:bg-primary-600 transition-colors"
                    >
                        Back to Authors
                    </button>
                </div>
            </div>
        );
    }

    const toggleFollow = () => {
        setAuthor((prev: any) => ({ ...prev, isFollowing: !prev.isFollowing }));
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-6xl mx-auto px-4 py-6">


                    {/* Author Info */}
                    <div className="flex items-start gap-6">
                        <div className="w-24 h-24 bg-primary-500 text-white rounded-full flex items-center justify-center text-3xl font-semibold flex-shrink-0">
                            {author.avatar}
                        </div>
                        <div className="flex-1 min-w-0">
                            <h2 className="text-3xl font-bold text-gray-900 mb-2">{author.name}</h2>
                            <p className="text-xl text-gray-600 mb-1">{author.title}</p>
                            <p className="text-lg text-gray-500 mb-3">{author.institution}</p>
                            
                            <div className="flex items-center gap-4 mb-4">
                                <div className="flex items-center gap-6 text-sm text-gray-600">
                                    <div>
                                        <span className="font-semibold text-gray-900 text-lg">{author.hIndex}</span>
                                        <span className="ml-1">h-index</span>
                                    </div>
                                    <div>
                                        <span className="font-semibold text-gray-900 text-lg">{author.totalCitations.toLocaleString()}</span>
                                        <span className="ml-1">citations</span>
                                    </div>
                                    <div>
                                        <span className="font-semibold text-gray-900 text-lg">{author.papers.length}</span>
                                        <span className="ml-1">papers</span>
                                    </div>
                                </div>
                                <button
                                    onClick={toggleFollow}
                                    className={`px-6 py-2 rounded-full font-medium transition-colors ${
                                        author.isFollowing
                                            ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                            : 'bg-primary-500 text-white hover:bg-primary-600'
                                    }`}
                                >
                                    {author.isFollowing ? 'Following' : 'Follow'}
                                </button>
                                <span
                                    className={`px-3 py-1 rounded-full text-sm font-semibold ${
                                        author.claimed
                                            ? 'bg-green-100 text-green-700 border border-green-200'
                                            : 'bg-gray-100 text-gray-600 border border-gray-200'
                                    }`}
                                >
                                    {author.claimed ? 'Claimed Profile' : 'Unclaimed Profile'}
                                </span>
                            </div>

                            <div className="flex flex-wrap gap-2">
                                {author.expertise.map((exp: string, index: number) => (
                                    <span
                                        key={index}
                                        className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded-full"
                                    >
                                        {exp}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-6xl mx-auto px-4">
                    <div className="flex space-x-8">
                        {[
                            { id: 'overview', label: 'Overview' },
                            { id: 'papers', label: `Papers (${authorPapers.length})` },
                            { id: 'network', label: 'Network' }
                        ].map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id as any)}
                                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                                    activeTab === tab.id
                                        ? 'border-primary-500 text-primary-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                }`}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="max-w-6xl mx-auto px-4 py-6">
                {activeTab === 'overview' && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2">
                            <div className="bg-white rounded-lg border border-gray-200 p-6">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Papers</h3>
                                <div className="space-y-4">
                                    {author.papers.slice(0, 5).map((paper: any) => (
                                        <div key={paper.id} className="border-b border-gray-100 pb-4 last:border-b-0">
                                            <h4 className="font-medium text-gray-900 mb-1">{paper.title}</h4>
                                            <p className="text-sm text-gray-600 mb-2">{paper.year} • {paper.citations} citations</p>
                                            <a
                                                href={paper.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-sm text-primary-500 hover:text-primary-600 underline"
                                            >
                                                View Paper
                                            </a>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                        <div className="space-y-6">
                            <div className="bg-white rounded-lg border border-gray-200 p-6">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Affiliations</h3>
                                <div className="space-y-3">
                                    {authorPapers.map(paper => 
                                        paper.affiliations.map((aff: any) => {
                                            const affiliation = mockAffiliations.find(a => a.id === aff.id);
                                            return affiliation ? (
                                                <div key={aff.id} className="flex items-center gap-3">
                                                    <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-semibold">
                                                        {affiliation.initials}
                                                    </div>
                                                    <div>
                                                        <p className="text-sm font-medium text-gray-900">{affiliation.label}</p>
                                                        <p className="text-xs text-gray-500">{affiliation.location}</p>
                                                    </div>
                                                </div>
                                            ) : null;
                                        })
                                    ).flat().filter((v, i, a) => a.indexOf(v) === i)}
                                </div>
                            </div>
                            <div className="bg-white rounded-lg border border-gray-200 p-6">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Activity</h3>
                                <p className="text-sm text-gray-600">Last active {author.recentActivity}</p>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'papers' && (
                    <div className="bg-white rounded-lg border border-gray-200">
                        <div className="p-6 border-b border-gray-200">
                            <h3 className="text-lg font-semibold text-gray-900">All Papers</h3>
                        </div>
                        <div className="divide-y divide-gray-200">
                            {authorPapers.map((paper, index) => (
                                <div key={paper.id} className="p-6">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <h4 className="text-lg font-medium text-gray-900 mb-2">{paper.title}</h4>
                                            <div className="flex items-center gap-4 text-sm text-gray-600 mb-3">
                                                <span>{paper.year}</span>
                                                <span>•</span>
                                                <span>{paper.citations} citations</span>
                                                <span>•</span>
                                                <span>{paper.stars} stars</span>
                                            </div>
                                            <div className="flex flex-wrap gap-2 mb-3">
                                                {paper.tags.map((tag: string, idx: number) => (
                                                    <span key={idx} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
                                                        {tag}
                                                    </span>
                                                ))}
                                            </div>
                                            <div className="flex items-center gap-4 text-sm text-gray-600">
                                                <span>Affiliations: {paper.affiliations.map((aff: any) => aff.label).join(', ')}</span>
                                            </div>
                                        </div>
                                        <a
                                            href={paper.link}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="ml-4 text-primary-500 hover:text-primary-600"
                                        >
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 3h7v7m0 0L10 21l-7-7 11-11z" />
                                            </svg>
                                        </a>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {activeTab === 'network' && (
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Collaboration Network</h3>
                        <p className="text-gray-600">Network visualization coming soon...</p>
                    </div>
                )}
            </div>
        </div>
    );
} 