import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { mockAffiliations } from './mockData/affiliations';
import { mockScholars } from './mockData/scholars';
import { mockPapers } from './mockData/papers';

interface AffiliationProfileProps {
    repo?: unknown;
    affiliation?: any;
    onClose?: () => void;
}

export function AffiliationProfile({ repo: _repo, affiliation: propAffiliation, onClose }: AffiliationProfileProps) {
    const { affiliationId } = useParams<{ affiliationId: string }>();
    const navigate = useNavigate();
    const [affiliation, setAffiliation] = useState<any>(propAffiliation || null);
    const [affiliationMembers, setAffiliationMembers] = useState<any[]>([]);
    const [affiliationPapers, setAffiliationPapers] = useState<any[]>([]);
    const [activeTab, setActiveTab] = useState<'overview' | 'members' | 'papers'>('overview');

    useEffect(() => {
        if (propAffiliation) {
            setAffiliation(propAffiliation);
            // Find scholars associated with this affiliation
            const members = mockScholars.filter(scholar => 
                scholar.institution.toLowerCase().includes(propAffiliation.name.toLowerCase()) ||
                scholar.institution.toLowerCase().includes(propAffiliation.label.toLowerCase())
            );
            setAffiliationMembers(members);
            // Find papers from this affiliation
            const papers = mockPapers.filter(paper => 
                paper.affiliations.some((aff: any) => aff.id === propAffiliation.id)
            );
            setAffiliationPapers(papers);
        } else if (affiliationId) {
            const foundAffiliation = mockAffiliations.find(a => a.id === affiliationId);
            if (foundAffiliation) {
                setAffiliation(foundAffiliation);
                const members = mockScholars.filter(scholar => 
                    scholar.institution.toLowerCase().includes(foundAffiliation.name.toLowerCase()) ||
                    scholar.institution.toLowerCase().includes(foundAffiliation.label.toLowerCase())
                );
                setAffiliationMembers(members);
                const papers = mockPapers.filter(paper => 
                    paper.affiliations.some((aff: any) => aff.id === affiliationId)
                );
                setAffiliationPapers(papers);
            }
        }
    }, [propAffiliation, affiliationId]);

    if (!affiliation) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 mb-2">Affiliation not found</h1>
                    <p className="text-gray-600 mb-4">The affiliation you're looking for doesn't exist.</p>
                    <button 
                        onClick={() => navigate('/affiliations')}
                        className="bg-primary-500 text-white px-4 py-2 rounded-lg hover:bg-primary-600 transition-colors"
                    >
                        Back to Affiliations
                    </button>
                </div>
            </div>
        );
    }



    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-6xl mx-auto px-4 py-6">


                    {/* Affiliation Info */}
                    <div className="flex items-start gap-6">
                        <div className="w-24 h-24 bg-primary-500 text-white rounded-full flex items-center justify-center text-3xl font-semibold flex-shrink-0">
                            {affiliation.initials}
                        </div>
                        <div className="flex-1 min-w-0">
                            <h2 className="text-3xl font-bold text-gray-900 mb-2">{affiliation.label}</h2>
                            <p className="text-xl text-gray-600 mb-1">{affiliation.name}</p>
                            <p className="text-lg text-gray-500 mb-3">{affiliation.location}</p>
                            
                            <div className="flex items-center gap-4 mb-4">
                                <div className="flex items-center gap-6 text-sm text-gray-600">
                                    <div>
                                        <span className="font-semibold text-gray-900 text-lg">{affiliation.memberCount}</span>
                                        <span className="ml-1">members</span>
                                    </div>
                                    <div>
                                        <span className="font-semibold text-gray-900 text-lg">{affiliation.papers}</span>
                                        <span className="ml-1">papers</span>
                                    </div>
                                    <div>
                                        <span className="font-semibold text-gray-900 text-lg">{affiliation.citations.toLocaleString()}</span>
                                        <span className="ml-1">citations</span>
                                    </div>
                                </div>

                                <span
                                    className={`px-3 py-1 rounded-full text-sm font-semibold ${
                                        affiliation.isAdmin
                                            ? 'bg-blue-100 text-blue-700 border border-blue-200'
                                            : 'bg-gray-100 text-gray-600 border border-gray-200'
                                    }`}
                                >
                                    {affiliation.type}
                                </span>
                            </div>

                            <div className="flex flex-wrap gap-2">
                                {affiliation.tags.map((tag: string, index: number) => (
                                    <span
                                        key={index}
                                        className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded-full"
                                    >
                                        {tag}
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
                            { id: 'members', label: `Members (${affiliationMembers.length})` },
                            { id: 'papers', label: `Papers (${affiliationPapers.length})` }
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
                                    {affiliationPapers.slice(0, 5).map((paper: any) => (
                                        <div key={paper.id} className="border-b border-gray-100 pb-4 last:border-b-0">
                                            <h4 className="font-medium text-gray-900 mb-1">{paper.title}</h4>
                                            <p className="text-sm text-gray-600 mb-2">
                                                {paper.authors.map((authorId: string, idx: number) => {
                                                    const author = mockScholars.find(s => s.id === authorId);
                                                    return author ? (
                                                        <span key={author.id}>
                                                            {author.name}{idx < paper.authors.length - 1 ? ', ' : ''}
                                                        </span>
                                                    ) : null;
                                                })}
                                            </p>
                                            <p className="text-sm text-gray-500 mb-2">{paper.year} • {paper.citations} citations</p>
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
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Members</h3>
                                <div className="space-y-3">
                                    {affiliationMembers.slice(0, 5).map((member: any) => (
                                        <div key={member.id} className="flex items-center gap-3">
                                            <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-semibold">
                                                {member.avatar}
                                            </div>
                                            <div>
                                                <p className="text-sm font-medium text-gray-900">{member.name}</p>
                                                <p className="text-xs text-gray-500">{member.title}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            
                            <div className="bg-white rounded-lg border border-gray-200 p-6">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Contact</h3>
                                <div className="space-y-2">
                                    <p className="text-sm text-gray-600">
                                        <span className="font-medium">Location:</span> {affiliation.location}
                                    </p>
                                    <p className="text-sm text-gray-600">
                                        <span className="font-medium">Type:</span> {affiliation.type}
                                    </p>
                                    <p className="text-sm text-gray-600">
                                        <span className="font-medium">Last Active:</span> {affiliation.lastActive}
                                    </p>
                                    <a
                                        href={affiliation.website}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-sm text-primary-500 hover:text-primary-600 underline block"
                                    >
                                        Visit Website
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'members' && (
                    <div className="bg-white rounded-lg border border-gray-200">
                        <div className="p-6">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4">All Members</h3>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                {affiliationMembers.map((member: any) => (
                                    <div key={member.id} className="flex items-center gap-3 p-3 rounded-lg border border-gray-100 hover:bg-gray-50">
                                        <div className="w-10 h-10 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-semibold">
                                            {member.avatar}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-gray-900 truncate">{member.name}</p>
                                            <p className="text-xs text-gray-500 truncate">{member.title}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'papers' && (
                    <div className="bg-white rounded-lg border border-gray-200">
                        <div className="p-6">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4">All Papers</h3>
                            <div className="space-y-4">
                                {affiliationPapers.map((paper: any) => (
                                    <div key={paper.id} className="border-b border-gray-100 pb-4 last:border-b-0">
                                        <h4 className="font-medium text-gray-900 mb-1">{paper.title}</h4>
                                        <p className="text-sm text-gray-600 mb-2">
                                            {paper.authors.map((authorId: string, idx: number) => {
                                                const author = mockScholars.find(s => s.id === authorId);
                                                return author ? (
                                                    <span key={author.id}>
                                                        {author.name}{idx < paper.authors.length - 1 ? ', ' : ''}
                                                    </span>
                                                ) : null;
                                            })}
                                        </p>
                                        <p className="text-sm text-gray-500 mb-2">{paper.year} • {paper.citations} citations</p>
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
                )}
            </div>
        </div>
    );
} 