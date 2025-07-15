import React, { useState, useEffect } from 'react';
import { PaperCard } from '../cards';

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

// Type definitions for feed posts
interface Post {
    id: number;
    name?: string;
    username?: string;
    avatar?: string;
    content?: string;
    time?: string;
    likes?: number;
    retweets?: number;
    replies?: number;
    type: 'user-post' | 'paper-post' | 'pure-paper';
    paper?: {
        id: string;
        title: string;
        author: string;
        authorInitial: string;
        summary: string;
        abstract: string;
        citations: number;
        url: string;
    };
}

interface FeedViewProps {
    // Props for MathJax integration
    mathJaxLoaded: boolean;
    postsRef: React.RefObject<HTMLDivElement | null>;
    // Callback for when posts change (for MathJax re-rendering)
    onPostsChange?: () => void;
}

/**
 * FeedView Component
 * 
 * This component handles the social feed functionality of the Library app, including:
 * - Post composition with LaTeX support
 * - Feed display with different post types (user posts, paper posts, pure papers)
 * - MathJax integration for mathematical content rendering
 * - Interactive elements like likes, retweets, and replies
 * 
 * The component is designed to be self-contained while integrating with the parent
 * Library component for MathJax functionality and post state management.
 */
export function FeedView({ mathJaxLoaded, postsRef, onPostsChange }: FeedViewProps) {
    // State for post composition
    const [composerText, setComposerText] = useState('');
    
    // State for expanded abstracts (which papers have their abstracts visible)
    const [expandedAbstracts, setExpandedAbstracts] = useState<Set<string>>(new Set());

    // Dummy posts data with mathematical content
    // In a real application, this would come from props or an API
    const posts: Post[] = [
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
    ];

    /**
     * Handles the submission of a new post
     * In a real application, this would send the post to an API
     */
    const handlePostSubmit = () => {
        if (!composerText.trim()) return;
        
        // Here you would typically:
        // 1. Send the post to your backend API
        // 2. Update the posts state with the new post
        // 3. Clear the composer text
        // 4. Trigger MathJax re-rendering
        
        console.log('Submitting post:', composerText);
        setComposerText('');
        
        // Notify parent component that posts have changed (for MathJax re-rendering)
        if (onPostsChange) {
            onPostsChange();
        }
    };

    /**
     * Toggles the visibility of a paper's abstract
     * @param paperId - The unique identifier of the paper
     */
    const toggleAbstract = (paperId: string) => {
        setExpandedAbstracts(prev => {
            const newSet = new Set(prev);
            if (newSet.has(paperId)) {
                newSet.delete(paperId);
            } else {
                newSet.add(paperId);
            }
            return newSet;
        });
    };

    /**
     * Renders the post composer section where users can create new posts
     */
    const renderPostComposer = () => (
        <div className="bg-white border-b border-gray-200 p-6">
            <div className="flex gap-4">
                {/* User avatar */}
                <div className="w-10 h-10 bg-primary-500 text-white rounded-full flex items-center justify-center font-semibold text-sm flex-shrink-0">
                    U
                </div>
                
                {/* Post composition area */}
                <div className="flex-1">
                    <textarea
                        className="w-full min-h-[80px] max-h-32 border-0 resize-none focus:ring-0 focus:outline-none text-gray-900 placeholder-gray-500"
                        placeholder="Share your research insights… Include arXiv URLs to automatically import papers. LaTeX supported with $...$ or $$...$$"
                        value={composerText}
                        onChange={(e) => setComposerText(e.target.value)}
                        onKeyDown={(e) => {
                            // Submit on Enter (without Shift for new line)
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handlePostSubmit();
                            }
                        }}
                    />
                    
                    {/* Post button */}
                    <div className="flex justify-end mt-3">
                        <button
                            className={`px-6 py-2 rounded-full font-medium transition-colors ${
                                composerText.trim()
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
    );

    /**
     * Renders a single post in the feed
     * @param post - The post object to render
     */
    const renderPost = (post: Post) => {
        // Handle pure paper posts (papers without user commentary)
        if (post.type === 'pure-paper') {
            return (
                <div key={post.id} className="bg-white border-b border-gray-200 p-6">
                    <PaperCard 
                        paper={post.paper!} 
                        expandedAbstracts={expandedAbstracts}
                        onToggleAbstract={toggleAbstract}
                    />
                </div>
            );
        }

        // Handle user posts and paper posts with commentary
        return (
            <div 
                key={post.id} 
                className={`border-b border-gray-200 p-6 ${
                    post.type === 'user-post' ? 'bg-blue-50' :
                    post.type === 'paper-post' ? 'bg-orange-50' : 'bg-white'
                }`}
            >
                {/* Post header with user info */}
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

                {/* Post content */}
                <div className="text-gray-900 leading-relaxed mb-4 whitespace-pre-wrap">
                    {post.content}
                </div>

                {/* Embedded paper card for paper posts */}
                {post.type === 'paper-post' && post.paper && (
                    <PaperCard 
                        paper={post.paper} 
                        expandedAbstracts={expandedAbstracts}
                        onToggleAbstract={toggleAbstract}
                    />
                )}

                {/* Post interaction buttons */}
                <div className="flex items-center gap-6 text-gray-500">
                    {/* Reply button */}
                    <button className="flex items-center gap-2 hover:text-gray-700 transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                        <span className="text-sm">{post.replies || 0}</span>
                    </button>
                    
                    {/* Retweet button */}
                    <button className="flex items-center gap-2 hover:text-gray-700 transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                        </svg>
                        <span className="text-sm">{post.retweets || 0}</span>
                    </button>
                    
                    {/* Like button */}
                    <button className="flex items-center gap-2 hover:text-red-500 transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                        <span className="text-sm">{post.likes || 0}</span>
                    </button>
                    
                    {/* Share button */}
                    <button className="flex items-center gap-2 hover:text-gray-700 transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
                        </svg>
                    </button>
                </div>
            </div>
        );
    };

    /**
     * Renders the main feed list
     */
    const renderFeedList = () => (
        <div className="max-w-2xl mx-auto" ref={postsRef}>
            {posts.map(renderPost)}
        </div>
    );

    // MathJax rendering effect
    useEffect(() => {
        if (mathJaxLoaded && postsRef.current) {
            const renderMath = () => {
                const mathJax = window.MathJax;
                const postsElement = postsRef.current;
                if (mathJax?.typesetPromise && postsElement) {
                    mathJax.typesetPromise([postsElement]).then(() => {
                        console.log('FeedView MathJax rendering completed successfully');
                    }).catch((err: any) => {
                        console.error('FeedView MathJax error:', err);
                    });
                }
            };

            // Try immediately and after delays to ensure DOM is ready
            renderMath();
            setTimeout(renderMath, 100);
            setTimeout(renderMath, 500);
        }
    }, [mathJaxLoaded, posts]); // Re-render when posts change

    return (
        <>
            {renderPostComposer()}
            {renderFeedList()}
        </>
    );
} 