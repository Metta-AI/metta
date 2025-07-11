import { mockUsers, User } from './users';
import { mockScholars, Scholar } from './scholars';
import { mockAffiliations, Affiliation } from './affiliations';

export interface Paper {
    id: string;
    title: string;
    starred: boolean;
    authors: { id: string; name: string }[];
    affiliations: { id: string; label: string }[];
    tags: string[];
    readBy: User[];
    queued: User[];
    link: string;
    stars: number;
}

const tagPool = [
    'Transformer', 'NLP', 'Deep Learning', 'VAE', 'Generative Models',
    'Reinforcement Learning', 'Policy Gradient', 'Robotics', 'Vision', 'Theory',
    'Large Language Models', 'AI Safety', 'Recommendation Systems', 'Computer Vision',
    'Approximation Algorithms', 'Control Theory', 'Audio AI', 'AR/VR', 'Document AI', 'Code AI'
];

const researchAreas = [
    'Attention Mechanisms', 'Transformer Architecture', 'Natural Language Processing',
    'Reinforcement Learning', 'Robotics', 'Control Theory', 'Neural Network Theory',
    'Approximation Algorithms', 'Theoretical Computer Science', 'Computer Vision',
    'Deep Learning', 'Convolutional Neural Networks', 'Generative Models',
    'Variational Methods', 'Large Language Models', 'Computer Graphics',
    'Machine Learning', 'AI Safety', 'Quantum Computing', 'Systems',
    'Audio AI', 'Recommendation Systems', 'Autonomous Vehicles', 'AR/VR',
    'Conversational AI', 'Document AI', 'Code AI', 'Question Answering'
];

const methods = [
    'Attention', 'Transformer', 'Convolutional', 'Recurrent', 'Graph Neural Networks',
    'Variational Autoencoders', 'Generative Adversarial Networks', 'Policy Gradients',
    'Actor-Critic', 'Monte Carlo', 'Bayesian', 'Federated Learning', 'Meta-Learning',
    'Self-Supervised Learning', 'Contrastive Learning', 'Knowledge Distillation',
    'Neural Architecture Search', 'AutoML', 'Multi-Task Learning', 'Transfer Learning'
];

const domains = [
    'Language Modeling', 'Machine Translation', 'Question Answering', 'Text Generation',
    'Image Classification', 'Object Detection', 'Semantic Segmentation', 'Image Generation',
    'Video Understanding', 'Audio Processing', 'Speech Recognition', 'Music Generation',
    'Robotics Control', 'Autonomous Driving', 'Game Playing', 'Recommendation Systems',
    'Drug Discovery', 'Medical Imaging', 'Climate Modeling', 'Financial Forecasting',
    'Code Generation', 'Document Analysis', 'Social Media Analysis', 'Scientific Computing'
];

const titleTemplates = [
    '{method} for {domain}: A {approach} Approach',
    '{method}-Based {domain} with {approach}',
    'Improving {domain} through {method} and {approach}',
    '{method} Meets {domain}: {approach} Solutions',
    'Towards Better {domain} with {method}',
    '{method} in {domain}: {approach} Perspectives',
    'Efficient {method} for {domain}',
    'Robust {method} for {domain} Applications',
    'Scalable {method} for {domain}',
    '{method} and {domain}: {approach} Methods',
    'Advanced {method} Techniques for {domain}',
    '{method} Optimization for {domain}',
    'Novel {method} Approaches to {domain}',
    '{method} for Real-World {domain}',
    'State-of-the-Art {method} in {domain}'
];

const approaches = [
    'Neural', 'Deep Learning', 'Machine Learning', 'Statistical', 'Probabilistic',
    'Optimization-Based', 'Heuristic', 'Analytical', 'Computational', 'Hybrid',
    'Multi-Modal', 'Cross-Domain', 'Adaptive', 'Dynamic', 'Hierarchical',
    'Attention-Based', 'Memory-Augmented', 'Knowledge-Enhanced', 'Context-Aware'
];

function getRandomElement<T>(arr: T[]): T {
    return arr[Math.floor(Math.random() * arr.length)];
}

function getRandomElements<T>(arr: T[], count: number): T[] {
    const shuffled = arr.slice().sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}

function generateRealisticTitle(): string {
    const template = getRandomElement(titleTemplates);
    const method = getRandomElement(methods);
    const domain = getRandomElement(domains);
    const approach = getRandomElement(approaches);
    
    return template
        .replace('{method}', method)
        .replace('{domain}', domain)
        .replace('{approach}', approach);
}

export const mockPapers: Paper[] = Array.from({ length: 100 }, (_, i) => {
    const authors = getRandomElements(mockScholars, Math.floor(Math.random() * 3) + 1).map(s => ({ id: s.id, name: s.name }));
    const affiliations = getRandomElements(mockAffiliations, Math.floor(Math.random() * 2) + 1).map(a => ({ id: a.id, label: a.label }));
    const tags = getRandomElements(tagPool, Math.floor(Math.random() * 3) + 1);
    const readBy = getRandomElements(mockUsers, Math.floor(Math.random() * 5));
    const queued = getRandomElements(mockUsers, Math.floor(Math.random() * 3));
    return {
        id: `p${i + 1}`,
        title: generateRealisticTitle(),
        starred: Math.random() > 0.8,
        authors,
        affiliations,
        tags,
        readBy,
        queued,
        link: `https://arxiv.org/abs/mock${i + 1}`,
        stars: Math.floor(Math.random() * 20),
    };
});

mockPapers.push(...Array.from({ length: 100 }, (_, i) => {
    const idx = i + 100;
    const authors = getRandomElements(mockScholars, Math.floor(Math.random() * 3) + 1).map(s => ({ id: s.id, name: s.name }));
    const affiliations = getRandomElements(mockAffiliations, Math.floor(Math.random() * 2) + 1).map(a => ({ id: a.id, label: a.label }));
    const tags = getRandomElements(tagPool, Math.floor(Math.random() * 3) + 1);
    const readBy = getRandomElements(mockUsers, Math.floor(Math.random() * 5));
    const queued = getRandomElements(mockUsers, Math.floor(Math.random() * 3));
    return {
        id: `p${idx + 1}`,
        title: generateRealisticTitle(),
        starred: Math.random() > 0.8,
        authors,
        affiliations,
        tags,
        readBy,
        queued,
        link: `https://arxiv.org/abs/mock${idx + 1}`,
        stars: Math.floor(Math.random() * 20),
    };
}));

mockPapers.push(...Array.from({ length: 100 }, (_, i) => {
    const idx = i + 200;
    const authors = getRandomElements(mockScholars, Math.floor(Math.random() * 3) + 1).map(s => ({ id: s.id, name: s.name }));
    const affiliations = getRandomElements(mockAffiliations, Math.floor(Math.random() * 2) + 1).map(a => ({ id: a.id, label: a.label }));
    const tags = getRandomElements(tagPool, Math.floor(Math.random() * 3) + 1);
    const readBy = getRandomElements(mockUsers, Math.floor(Math.random() * 5));
    const queued = getRandomElements(mockUsers, Math.floor(Math.random() * 3));
    return {
        id: `p${idx + 1}`,
        title: generateRealisticTitle(),
        starred: Math.random() > 0.8,
        authors,
        affiliations,
        tags,
        readBy,
        queued,
        link: `https://arxiv.org/abs/mock${idx + 1}`,
        stars: Math.floor(Math.random() * 20),
    };
}));

mockPapers.push(...Array.from({ length: 100 }, (_, i) => {
    const idx = i + 300;
    const authors = getRandomElements(mockScholars, Math.floor(Math.random() * 3) + 1).map(s => ({ id: s.id, name: s.name }));
    const affiliations = getRandomElements(mockAffiliations, Math.floor(Math.random() * 2) + 1).map(a => ({ id: a.id, label: a.label }));
    const tags = getRandomElements(tagPool, Math.floor(Math.random() * 3) + 1);
    const readBy = getRandomElements(mockUsers, Math.floor(Math.random() * 5));
    const queued = getRandomElements(mockUsers, Math.floor(Math.random() * 3));
    return {
        id: `p${idx + 1}`,
        title: generateRealisticTitle(),
        starred: Math.random() > 0.8,
        authors,
        affiliations,
        tags,
        readBy,
        queued,
        link: `https://arxiv.org/abs/mock${idx + 1}`,
        stars: Math.floor(Math.random() * 20),
    };
})); 