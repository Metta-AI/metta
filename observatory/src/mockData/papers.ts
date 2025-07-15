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

export const mockPapers = [
  {
    id: "paper1",
    title: "Conversational AI for Robotics",
    year: 2022,
    citations: 500,
    authors: ["alex-abbott", "bella-baker"], // Using actual scholar IDs
    affiliations: ["stanford-ai-lab"], // Using actual affiliation IDs
    tags: ["Conversational AI", "Robotics"],
    link: "https://arxiv.org/abs/1234.5678",
    readBy: [],
    queued: [],
    starred: false,
    stars: 10,
  },
  {
    id: "paper2",
    title: "Natural Language Processing in Autonomous Vehicles",
    year: 2021,
    citations: 700,
    authors: ["carla-chen"], // Using actual scholar IDs
    affiliations: ["berkeley-ai"], // Using actual affiliation IDs
    tags: ["NLP", "Autonomous Vehicles"],
    link: "https://arxiv.org/abs/2345.6789",
    readBy: [],
    queued: [],
    starred: false,
    stars: 8,
  },
  {
    id: "paper3",
    title: "Transformer Architecture for Computer Vision",
    year: 2023,
    citations: 1200,
    authors: ["daniel-diaz"], // Using actual scholar IDs
    affiliations: ["mit-csail"], // Using actual affiliation IDs
    tags: ["Transformer", "Computer Vision", "Deep Learning"],
    link: "https://arxiv.org/abs/3456.7890",
    readBy: [],
    queued: [],
    starred: true,
    stars: 25,
  },
  {
    id: "paper4",
    title: "Variational Autoencoders for Generative Models",
    year: 2022,
    citations: 850,
    authors: ["emma-evans"], // Using actual scholar IDs
    affiliations: ["openai"], // Using actual affiliation IDs
    tags: ["VAE", "Generative Models", "Deep Learning"],
    link: "https://arxiv.org/abs/4567.8901",
    readBy: [],
    queued: [],
    starred: false,
    stars: 15,
  },
  {
    id: "paper5",
    title: "Reinforcement Learning in Game Playing",
    year: 2021,
    citations: 950,
    authors: ["felix-foster", "grace-garcia"], // Using actual scholar IDs
    affiliations: ["cmu-ml", "google-brain"], // Using actual affiliation IDs
    tags: ["Reinforcement Learning", "Game Playing", "AI Safety"],
    link: "https://arxiv.org/abs/5678.9012",
    readBy: [],
    queued: [],
    starred: true,
    stars: 18,
  },
  {
    id: "paper6",
    title: "Attention Mechanisms for Large Language Models",
    year: 2023,
    citations: 1500,
    authors: ["hannah-hughes", "ian-irving"], // Using actual scholar IDs
    affiliations: ["oxford-ai", "facebook-ai"], // Using actual affiliation IDs
    tags: ["Attention Mechanisms", "Large Language Models", "NLP"],
    link: "https://arxiv.org/abs/6789.0123",
    readBy: [],
    queued: [],
    starred: false,
    stars: 22,
  },
  {
    id: "paper7",
    title: "Deep Learning for Computer Vision Applications",
    year: 2022,
    citations: 1100,
    authors: ["julia-jones"], // Using actual scholar IDs
    affiliations: ["toronto-ai"], // Using actual affiliation IDs
    tags: ["Deep Learning", "Computer Vision", "Neural Networks"],
    link: "https://arxiv.org/abs/7890.1234",
    readBy: [],
    queued: [],
    starred: true,
    stars: 30,
  },
  {
    id: "paper8",
    title: "AI Systems for Distributed Computing",
    year: 2023,
    citations: 800,
    authors: ["karen-kim", "liam-lee"], // Using actual scholar IDs
    affiliations: ["microsoft-research", "eth-zurich"], // Using actual affiliation IDs
    tags: ["AI Systems", "Distributed Computing", "Machine Learning Infrastructure"],
    link: "https://arxiv.org/abs/8901.2345",
    readBy: [],
    queued: [],
    starred: false,
    stars: 12,
  },
  {
    id: "paper9",
    title: "Neural Architecture Search for Efficient Models",
    year: 2023,
    citations: 950,
    authors: ["nina-nash", "oliver-owens"], // Using actual scholar IDs
    affiliations: ["mit-csail", "openai"], // Using actual affiliation IDs
    tags: ["Neural Architecture Search", "AutoML", "Model Optimization"],
    link: "https://arxiv.org/abs/9012.3456",
    readBy: [],
    queued: [],
    starred: false,
    stars: 16,
  },
  {
    id: "paper10",
    title: "Multimodal Learning for Vision-Language Tasks",
    year: 2022,
    citations: 1300,
    authors: ["paula-perez", "quincy-quinn", "rachel-ryan"], // Using actual scholar IDs
    affiliations: ["cmu-ml", "google-brain", "oxford-ai"], // Using actual affiliation IDs
    tags: ["Multimodal Learning", "Vision-Language", "Computer Vision"],
    link: "https://arxiv.org/abs/0123.4567",
    readBy: [],
    queued: [],
    starred: true,
    stars: 28,
  },
  {
    id: "paper11",
    title: "Robotic Learning for Autonomous Navigation",
    year: 2023,
    citations: 750,
    authors: ["steven-smith"], // Using actual scholar IDs
    affiliations: ["facebook-ai"], // Using actual affiliation IDs
    tags: ["Robotics", "Autonomous Navigation", "Machine Learning"],
    link: "https://arxiv.org/abs/1234.5679",
    readBy: [],
    queued: [],
    starred: false,
    stars: 14,
  },
  {
    id: "paper12",
    title: "Theoretical Foundations of Deep Learning",
    year: 2021,
    citations: 1800,
    authors: ["tina-tan", "ulrich-ulrich"], // Using actual scholar IDs
    affiliations: ["toronto-ai", "microsoft-research"], // Using actual affiliation IDs
    tags: ["Theoretical Machine Learning", "Deep Learning", "Optimization Theory"],
    link: "https://arxiv.org/abs/2345.6780",
    readBy: [],
    queued: [],
    starred: true,
    stars: 35,
  },
]; 