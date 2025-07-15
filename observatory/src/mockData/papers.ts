import { mockUsers, User } from '../src/mockData/users';
import { mockScholars, Scholar } from '../src/mockData/scholars';
import { mockInstitutions, Institution } from '../src/mockData/institutions';

export interface Paper {
    id: string;
    title: string;
    starred: boolean;
    authors: { id: string; name: string }[];
    institutions: { id: string; label: string }[];
    tags: string[];
    readBy: User[];
    queued: User[];
    link: string;
    stars: number;
}

// Generated from Asana migration on 2025-07-15T22:04:03.908Z
export const mockPapers: Paper[] = [
  {
    "id": "paper-1",
    "title": "High Dimensional Bayesian Optimization Assisted by Principal Component Analysis",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Optimization"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2007.00925",
    "stars": 0
  },
  {
    "id": "paper-2",
    "title": "Evolving Curricula with Regret-Based Environment Design",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2203.01302",
    "stars": 0
  },
  {
    "id": "paper-3",
    "title": "Evolution guides policy gradient in RL",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1805.07917",
    "stars": 0
  },
  {
    "id": "paper-4",
    "title": "How Weight Resampling and Optimizers Shape the Dynamics of Continual Learning and Forgetting in Neural Networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Continual Learning",
      "Plasticity"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2507.01559",
    "stars": 0
  },
  {
    "id": "paper-5",
    "title": "Entropy and Diversity: The Axiomatic Approach",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution",
      "Exploration",
      "Open Endedness"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2012.02113",
    "stars": 0
  },
  {
    "id": "paper-6",
    "title": "Associative conditioning in gene regulatory network models increases integrative causal emergence",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Memory",
      "Plasticity",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-7",
    "title": "Automatic Curriculum Learning For Deep RL: A Short Survey",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/2003.04664",
    "stars": 0
  },
  {
    "id": "paper-8",
    "title": "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning"
    ],
    "readBy": [
      {
        "id": "jack-heart",
        "name": "Jack Heart",
        "avatar": "JH",
        "email": "jack-heart@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2003.04960",
    "stars": 0
  },
  {
    "id": "paper-9",
    "title": "Contrastive Representation Learning (Blog Post)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Contrastive RL"
    ],
    "readBy": [
      {
        "id": "yudhister-kumar",
        "name": "Yudhister Kumar",
        "avatar": "YK",
        "email": "yudhister-kumar@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
    "stars": 3
  },
  {
    "id": "paper-10",
    "title": "Automated Curriculum Learning for Neural Networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning"
    ],
    "readBy": [
      {
        "id": "jack-heart",
        "name": "Jack Heart",
        "avatar": "JH",
        "email": "jack-heart@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1704.03003",
    "stars": 3
  },
  {
    "id": "paper-11",
    "title": "Curiosity-driven Exploration by Self-supervised Prediction",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "jack-heart",
        "name": "Jack Heart",
        "avatar": "JH",
        "email": "jack-heart@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/1705.05363",
    "stars": 0
  },
  {
    "id": "paper-12",
    "title": "Toward a theory of evolution as multilevel learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution"
    ],
    "readBy": [
      {
        "id": "jack-eicher",
        "name": "Jack Eicher",
        "avatar": "JE",
        "email": "jack-eicher@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://www.pnas.org/doi/full/10.1073/pnas.2120037119#sec-7",
    "stars": 0
  },
  {
    "id": "paper-13",
    "title": "Agent-based Modelling as a Service on Amazon EC2: Opportunities and Challenges",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-14",
    "title": "Hierarchical Learning for Generation with Long Source Sequences",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Neural Architecture Search"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2104.07545",
    "stars": 2
  },
  {
    "id": "paper-15",
    "title": "Searching For Activation Functions",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Neural Architecture Search"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1710.05941",
    "stars": 0
  },
  {
    "id": "paper-16",
    "title": "Follow up on \"A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals\"",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-17",
    "title": "Unsupervised pre-training in Biological Neural Networks.",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-18",
    "title": "Minimal Value-Equivalent Partial Models for Scalable and Robust Planning in Lifelong Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Continual Learning",
      "RL",
      "Trajectory Planning",
      "World Model"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2301.10119",
    "stars": 0
  },
  {
    "id": "paper-19",
    "title": "Provably Efficient Maximum Entropy Exploration",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1812.02690",
    "stars": 0
  },
  {
    "id": "paper-20",
    "title": "Follow up on \"1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities\"",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-21",
    "title": "A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Contrastive RL",
      "Exploration",
      "Goal Conditioned RL",
      "RL"
    ],
    "readBy": [
      {
        "id": "matthew-bull",
        "name": "Matthew Bull",
        "avatar": "MB",
        "email": "matthew-bull@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2408.05804",
    "stars": 0
  },
  {
    "id": "paper-22",
    "title": "1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Contrastive RL",
      "Goal Conditioned RL",
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2503.14858",
    "stars": 0
  },
  {
    "id": "paper-23",
    "title": "Contrastive Learning as Goal-Conditioned Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Goal Conditioned RL",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2206.07568",
    "stars": 0
  },
  {
    "id": "paper-24",
    "title": "Robustness and Plasticity in ActInf",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Plasticity"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2506.06897",
    "stars": 0
  },
  {
    "id": "paper-25",
    "title": "Deep Reinforcement Learning at the Edge of the Statistical Precipice",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "jack-eicher",
        "name": "Jack Eicher",
        "avatar": "JE",
        "email": "jack-eicher@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2108.13264",
    "stars": 0
  },
  {
    "id": "paper-26",
    "title": "Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Bayesian Mechanics",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.alphaxiv.org/abs/2502.06060",
    "stars": 0
  },
  {
    "id": "paper-27",
    "title": "Bayesian Filtering with Multiple Internal Models: Toward a Theory of Social Intelligence",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Agency and alignment",
      "Bayesian Mechanics",
      "Consciousness",
      "Meta-Learning",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://pubmed.ncbi.nlm.nih.gov/31614100/#:~:text=singing%29%2C%20under%20multiple%20generative%20models,Bayesian%20filtering%2C%20combined%20with%20model",
    "stars": 0
  },
  {
    "id": "paper-28",
    "title": "Model averaging, optimal inference, and habit formation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Agency and alignment",
      "Bayesian Mechanics",
      "Consciousness",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00457/full",
    "stars": 0
  },
  {
    "id": "paper-29",
    "title": "Hierarchical Active Inference: A Theory of Motivated Control",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Agency and alignment",
      "Bayesian Mechanics",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.unisr.it/mediaObject/unisr/psicologia/dottorato-neuroscienze-cognitive/document/Pezzulo-Rigoli-and-Friston-2018/original/Pezzulo+Rigoli+and+Friston+2018.pdf#:~:text=requires%20arbitration%20among%20multiple%20drives,1%E2%80%938%5D.%20Previous%20research%20has",
    "stars": 0
  },
  {
    "id": "paper-30",
    "title": "An Active Inference Approach to Modeling Structure Learning: Concept Learning as an Example Case",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Agency and alignment",
      "Consciousness",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00041/full",
    "stars": 0
  },
  {
    "id": "paper-31",
    "title": "An Introduction to Noncommutative Physics",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2305.03671",
    "stars": 0
  },
  {
    "id": "paper-32",
    "title": "Reconciling lambda-Returns with Experience Replay",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Optimization",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1810.09967",
    "stars": 0
  },
  {
    "id": "paper-33",
    "title": "Uncertainty Representations in State-Space Layers for Deep Reinforcement Learning under Partial Observability",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Memory",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2409.16824",
    "stars": 0
  },
  {
    "id": "paper-34",
    "title": "Energy-based Predictive Representations for Partially Observed Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://proceedings.mlr.press/v216/zhang23b.html",
    "stars": 0
  },
  {
    "id": "paper-35",
    "title": "General agents need world models",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      },
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2506.01622",
    "stars": 0
  },
  {
    "id": "paper-36",
    "title": "The Hijacker's Guide to biological systems:Manipulation by self‐defecting or foreign agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment",
      "Communication & Cooperation",
      "Plasticity"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://osf.io/preprints/osf/behxt_v3",
    "stars": 0
  },
  {
    "id": "paper-37",
    "title": "Harnessing the analog computing power of regulatory networks with the Regulatory Network Machine",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://pdf.sciencedirectassets.com/318494/AIP/1-s2.0-S2589004225007977/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDkepIT3cxFaUs1h%2BRpEWPo3QpLIL303%2B1frQ%2FZRNwmuwIgRXK6rTNzrfzFQYa0HedEi4otAon1BFV0OIr0s%2Brefz0qsgUIYRAFGgwwNTkwMDM1NDY4NjUiDAfehbnljR5OnwsvqiqPBXgh2l5SVAhIJkp2ssVGRjO8jM2GNVh1z8vxSgyWaOSZwPgaizsDQkYjWmIK5%2Fo68xmeUH%2FxTrjqKoCb%2FzeWIM9y1utmALkRdI7a0HhYR%2FKDgWZQZGhDDYyEYrrDnrJfALxvi0bshqyPppTunss38sqW5yxalfDWnR7G%2FfGXr9Ks0z0ryHOevGoslCSoB4LxexkuBviWo4ZKirKdokf%2BwRkJHZf9Jf5N5f8tlUCiXmlqQ2W3W5pamfAx0viQUWZho9sbtI%2Fu3KmFLBDPxCPztMn%2FGAnWNc6BICZHDt1jAiW8YaiEzuwHS3TY3WBXqbQYRU6W5kn85mahukidKjmQRxaGxhm25GYzMy8%2FlJKioGXzt9rFmO9NOhLygyejJ%2FT%2BkzfZCey74pjABYPB28S7xN9Zlkyho7vL8MsN8HfgJ5chVQrjOjyQUldBM7mvdVa3WWVLMSpoaNDCgvW%2BUUr1fBUnxFlLCkJxNfWVdZt%2Bn0P%2FtHV6N8vEmfvJk4HIFkUwbrj1TjiaW7SSUrGDGm9xsixZHV172vYDUdGGCyn2YD3eWBdjAwhg9fhreGd3THLsYdBcP4Wj9MgJijHAarY2btmc3GMzJAXvfRD99FTYGCvfi5h5OBdWbbcMfiqlefeA%2BnQlWMZkep8k%2FeveQEDUDrpJqpt2yNnn9EPM1iu7qvqTJupG%2B1%2FK3SDARIyp06KoNNU8h5fQ7WyEOR1reUv0PuGm9nLqjPOt8s9IgqxE1pDx3PvB9QvvjFPJ3KOCcjx3P1u4axa04MJGG%2F0ipwn%2FXbwbuTcdA8wXCCf1eb4ZLMuG94%2FuHYQwwKHMDaXuu6FkSqpgvrRMAiPf9z0xlERlc8brBxiFHnK3jYgjYmyFEZsw%2BpmMwgY6sQGSUnwGJQ%2FKwSZvJhbq9qfYBeew%2BY4k6Gr%2F5JERWdYugBt3sINzDHgebkvaxxlDRT0oClTT9j%2BFAAiUyAjDhOc%2F6QWFqjfb68vBSr298Oye8LNpGyPkTPg9aQ7IJ5Nh4Ygt%2B2sNk1qJHVHb4PmGsdvQPVl5K1yKuVewTVRY%2F39%2FFABETnsVVPZBb%2FCCAHY8nbjrGQzLbWTQQpQvq57C4XhspWlJDHB7eb%2Bxu%2BZlQf2FA5s%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250606T161019Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQE3HUGN7%2F20250606%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c2c317a4c1c83bbbe958370f64514e4ac354d6c0ab973c4c62154c7531f87262&hash=5e708de13e5e567f25fafba72f34c4d166886e8316471485fe969b1afdf12dd2&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2589004225007977&tid=spdf-e23d50cc-1703-4948-8414-4d0a74d27b92&sid=ad8fed4a2368814ecc194359ad46a6a133dagxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f1c5852065652000605&rr=94b92dddab0f174b&cc=us",
    "stars": 0
  },
  {
    "id": "paper-38",
    "title": "AI in a vat: Fundamental limits of efficient world modelling for agent sandboxing and interpretability",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Continual Learning",
      "World Model"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2504.04608",
    "stars": 0
  },
  {
    "id": "paper-39",
    "title": "Action and Perception as Divergence Minimization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Exploration",
      "Skill Discovery",
      "World Model"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2009.01791",
    "stars": 3
  },
  {
    "id": "paper-40",
    "title": "Latent Space Policies for Hierarchical Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Goal Conditioned RL",
      "RL",
      "Trajectory Planning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1804.02808?",
    "stars": 0
  },
  {
    "id": "paper-41",
    "title": "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2011.09533",
    "stars": 2
  },
  {
    "id": "paper-42",
    "title": "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2005.12729",
    "stars": 1
  },
  {
    "id": "paper-43",
    "title": "Complex harmonics reveal low-dimensional manifolds of critical brain dynamics",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation",
      "Consciousness",
      "Neural Architecture Search"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.kringelbach.org/papers/PRE_DecoKringelbach2025.pdf",
    "stars": 3
  },
  {
    "id": "paper-44",
    "title": "Batch size-invariance for policy optimization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2110.00641",
    "stars": 3
  },
  {
    "id": "paper-45",
    "title": "Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Evolution",
      "Memory",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://x.com/sporadicalia/status/1876695913059746297?s=12",
    "stars": 0
  },
  {
    "id": "paper-46",
    "title": "Deep Reinforcement Learning with Spiking Q-learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Memory",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/html/2201.09754v2",
    "stars": 0
  },
  {
    "id": "paper-47",
    "title": "Hyperbolic discounting outperforms in MARL",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://openreview.net/forum?id=O5iBlbX1Ih",
    "stars": 0
  },
  {
    "id": "paper-48",
    "title": "Spline theory for smoothing deep neural manifolds",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Meta-Learning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2502.07783",
    "stars": 0
  },
  {
    "id": "paper-49",
    "title": "ICLR: In-Context Learning of Representations",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "LLMs",
      "Memory",
      "Meta-Learning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2501.00070",
    "stars": 0
  },
  {
    "id": "paper-50",
    "title": "Multi-agent Dealer Market",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/1911.05892",
    "stars": 0
  },
  {
    "id": "paper-51",
    "title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework - Microsoft 2023 - Outperforms ChatGPT+Code Interpreter!",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment",
      "LLMs",
      "Meta-Learning",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.reddit.com/r/MachineLearning/comments/15xh3qb/r_autogen_enabling_nextgen_llm_applications_via/",
    "stars": 0
  },
  {
    "id": "paper-52",
    "title": "Gauge Theories and Fiber Bundles: Definitions, Pictures, and Results",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Topology"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://x.com/diracghost/status/1874575481007387094?s=46",
    "stars": 0
  },
  {
    "id": "paper-53",
    "title": "Density distribution in two Ising systems with particle exchange",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Evolution",
      "Memory",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://link.springer.com/article/10.1140/epjb/e2018-90045-5",
    "stars": 0
  },
  {
    "id": "paper-54",
    "title": "Grand canonical description of equilibrium and non-equilibrium systems using spin formalism",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Evolution",
      "Memory",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://hal.science/hal-02906059v1/document#:~:text=6,in%20the%20grand%20canonical%20description",
    "stars": 0
  },
  {
    "id": "paper-55",
    "title": "Thermodynamics of evolution and the origin of life",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Evolution",
      "Memory",
      "Thermodynamics"
    ],
    "readBy": [
      {
        "id": "jack-eicher",
        "name": "Jack Eicher",
        "avatar": "JE",
        "email": "jack-eicher@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://www.pnas.org/doi/full/10.1073/pnas.2120042119",
    "stars": 0
  },
  {
    "id": "paper-56",
    "title": "Numerically “exact” approach to open quantum dynamics: The hierarchical equations of motion (HEOM)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Evolution",
      "Memory",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://pubs.aip.org/aip/jcp/article/153/2/020901/76291/Numerically-exact-approach-to-open-quantum",
    "stars": 0
  },
  {
    "id": "paper-57",
    "title": "Generative Agents: Interactive Simulacra of Human Behavior",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "LLMs"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2304.03442",
    "stars": 0
  },
  {
    "id": "paper-58",
    "title": "PerceiverIO",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Transformer"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2107.14795",
    "stars": 0
  },
  {
    "id": "paper-59",
    "title": "Perceiver",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Transformer"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2103.03206",
    "stars": 0
  },
  {
    "id": "paper-60",
    "title": "Coordination Among Neural Modules Through a Shared Global Workspace",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2103.01197",
    "stars": 3
  },
  {
    "id": "paper-61",
    "title": "Learning Progress Driven Multi-Agent Curriculum",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Multi-Agency",
      "RL"
    ],
    "readBy": [
      {
        "id": "jack-heart",
        "name": "Jack Heart",
        "avatar": "JH",
        "email": "jack-heart@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2205.10016",
    "stars": 0
  },
  {
    "id": "paper-62",
    "title": "Continual Reinforcement Learning via Autoencoder-Driven Task and New Environment Recognition",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Continual Learning"
    ],
    "readBy": [
      {
        "id": "manuel-razo-mejia",
        "name": "Manuel Razo-Mejia",
        "avatar": "MR",
        "email": "manuel-razo-mejia@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2505.09003",
    "stars": 3
  },
  {
    "id": "paper-63",
    "title": "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2404.02258",
    "stars": 3
  },
  {
    "id": "paper-64",
    "title": "Consistent Dropout for Policy Gradient Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation",
      "Plasticity",
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2202.11818",
    "stars": 2
  },
  {
    "id": "paper-65",
    "title": "An Investigation into Pre-Training Object-Centric Representations for Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Robust Agents"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2302.04419",
    "stars": 3
  },
  {
    "id": "paper-66",
    "title": "In-Context Reinforcement Learning for Variable Action Spaces",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2312.13327",
    "stars": 3
  },
  {
    "id": "paper-67",
    "title": "Multi-Object Representation Learning with Iterative Variational Inference",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1903.00450",
    "stars": 2
  },
  {
    "id": "paper-68",
    "title": "Object-Centric Learning with Slot Attention",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2006.15055",
    "stars": 3
  },
  {
    "id": "paper-69",
    "title": "Continuous Thought Machines",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "dominik-farr",
        "name": "Dominik Farr",
        "avatar": "DF",
        "email": "dominik-farr@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2505.05522",
    "stars": 0
  },
  {
    "id": "paper-70",
    "title": "A Variational Synthesis of Evolutionary and Developmental Dynamics",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Death / Rebirth",
      "Evolution"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.mdpi.com/1099-4300/25/7/964",
    "stars": 0
  },
  {
    "id": "paper-71",
    "title": "Emergence of Goal-Directed Behaviors via Active Inference with Self-Prior",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2504.11075?",
    "stars": 0
  },
  {
    "id": "paper-72",
    "title": "Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Multi-Agency",
      "Robust Agents"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/html/2504.12714v2",
    "stars": 0
  },
  {
    "id": "paper-73",
    "title": "Evolution of Cooperation in LLM-Agent Societies: A Preliminary Study Using Different Punishment Strategies",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Multi-Agency",
      "Robust Agents",
      "World Model"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/html/2504.19487v1",
    "stars": 0
  },
  {
    "id": "paper-74",
    "title": "From motor control to team play in simulated humanoid football",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.science.org/doi/10.1126/scirobotics.abo0235",
    "stars": 0
  },
  {
    "id": "paper-75",
    "title": "Human-level performance in 3D multiplayer games with population-based reinforcement learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.science.org/doi/10.1126/science.aau6249",
    "stars": 0
  },
  {
    "id": "paper-76",
    "title": "A Generalized Training Approach for Multiagent Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization",
      "Robust Agents"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1909.12823",
    "stars": 0
  },
  {
    "id": "paper-77",
    "title": "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html",
    "stars": 0
  },
  {
    "id": "paper-78",
    "title": "Navigating the landscape of multiplayer games",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Multi-Agency",
      "Open Enededness",
      "Optimization",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.nature.com/articles/s41467-020-19244-4",
    "stars": 0
  },
  {
    "id": "paper-79",
    "title": "Heterogeneous Social Value Orientation Leads to Meaningful Diversity in Sequential Social Dilemmas",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Eval",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2305.00768",
    "stars": 0
  },
  {
    "id": "paper-80",
    "title": "Re-evaluating Evaluation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization",
      "Eval",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://proceedings.neurips.cc/paper_files/paper/2018/file/cdf1035c34ec380218a8cc9a43d438f9-Paper.pdf",
    "stars": 0
  },
  {
    "id": "paper-81",
    "title": "Real World Games Look Like Spinning Tops",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Multi-Agency",
      "Eval",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://proceedings.neurips.cc/paper/2020/file/ca172e964907a97d5ebd876bfdd4adbd-Paper.pdf",
    "stars": 0
  },
  {
    "id": "paper-82",
    "title": "α-Rank: Multi-Agent Evaluation by Evolution",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Multi-Agency",
      "Optimization",
      "Eval",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-83",
    "title": "Neural Replicator Dynamics",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Multi-Agency",
      "Optimization",
      "Robust Agents"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-84",
    "title": "Differentiable Game Mechanics",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://jmlr.csail.mit.edu/papers/volume20/19-008/19-008.pdf",
    "stars": 0
  },
  {
    "id": "paper-85",
    "title": "Empirical Game Theoretic Analysis: A Survey",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "LLMs",
      "Multi-Agency",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.jair.org/index.php/jair/article/view/16146",
    "stars": 0
  },
  {
    "id": "paper-86",
    "title": "One-Minute Video Generation with Test-Time Training",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2504.05298",
    "stars": 0
  },
  {
    "id": "paper-87",
    "title": "A Goal-centric Outlook on Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Goal Conditioned RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "robb-walters",
        "name": "Robb Walters",
        "avatar": "RW",
        "email": "robb-walters@softmax.ai"
      }
    ],
    "link": "https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(23)00207-3",
    "stars": 0
  },
  {
    "id": "paper-88",
    "title": "BAMDP Shaping: A Unified Framework for Intrinsic Motivation and Reward Shaping",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment",
      "Goal Conditioned RL",
      "Meta-Learning"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "robb-walters",
        "name": "Robb Walters",
        "avatar": "RW",
        "email": "robb-walters@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2409.05358",
    "stars": 0
  },
  {
    "id": "paper-89",
    "title": "Plasticity Loss in Deep RL: A Survey",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Continual Learning",
      "Plasticity",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2411.04832",
    "stars": 0
  },
  {
    "id": "paper-90",
    "title": "Parseval Regularization for Continual Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Open Enededness",
      "Optimization",
      "Plasticity",
      "RL",
      "Robust Agents"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/html/2412.07224v1",
    "stars": 0
  },
  {
    "id": "paper-91",
    "title": "Representation Learning with Contrastive Predictive Coding",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL",
      "World Model"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "jack-heart",
        "name": "Jack Heart",
        "avatar": "JH",
        "email": "jack-heart@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1807.03748",
    "stars": 3
  },
  {
    "id": "paper-92",
    "title": "Robust Agents Learn Causal World Model",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment",
      "Robust Agents",
      "World Model"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/2402.10877",
    "stars": 0
  },
  {
    "id": "paper-93",
    "title": "Decentralized Collective World Model for Emergent Communication and Coordination",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Communication & Cooperation",
      "Emergence",
      "World Model"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2504.03353",
    "stars": 0
  },
  {
    "id": "paper-94",
    "title": "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "ALife"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1706.02275",
    "stars": 0
  },
  {
    "id": "paper-95",
    "title": "Fractal Generative Models",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "LLMs",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2502.17437",
    "stars": 0
  },
  {
    "id": "paper-96",
    "title": "Inhibitory neurons for architecture search paper",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence",
      "Evolution",
      "Memory"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.pnas.org/doi/epdf/10.1073/pnas.2218173120",
    "stars": 0
  },
  {
    "id": "paper-97",
    "title": "Evolution Strategies as Scalable Alternative to Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution",
      "RL"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1703.03864",
    "stars": 0
  },
  {
    "id": "paper-98",
    "title": "Proximal Policy Optimization with Adaptive Exploration",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/html/2405.04664v1",
    "stars": 0
  },
  {
    "id": "paper-99",
    "title": "Efficient Exploration via State Marginal Matching",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1906.05274",
    "stars": 0
  },
  {
    "id": "paper-100",
    "title": "Bayesian Predictive Coding",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/html/2503.24016v1",
    "stars": 0
  },
  {
    "id": "paper-101",
    "title": "How To Scale Your (Transformer) Model",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://jax-ml.github.io/scaling-book/",
    "stars": 0
  },
  {
    "id": "paper-102",
    "title": "OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2503.17821",
    "stars": 0
  },
  {
    "id": "paper-103",
    "title": "Differentiable Logic Cellular Automata",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "NCA"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://google-research.github.io/self-organising-systems/difflogic-ca/",
    "stars": 0
  },
  {
    "id": "paper-104",
    "title": "An Enhanced Proximal Policy Optimization-Based Reinforcement Learning Method with Random Forest for Hyperparameter Optimization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Continual Learning",
      "Exploration",
      "Meta-Learning",
      "Optimization",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://www.mdpi.com/2076-3417/12/14/7006",
    "stars": 0
  },
  {
    "id": "paper-105",
    "title": "A Self-Tuning Actor-Critic Algorithm",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Meta-Learning",
      "Optimization",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2002.12928",
    "stars": 0
  },
  {
    "id": "paper-106",
    "title": "Emergent Reciprocity and Team Formation from Randomized Uncertain Social Preferences",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://proceedings.neurips.cc/paper/2020/file/b63c87b0a41016ad29313f0d7393cee8-Paper.pdf",
    "stars": 0
  },
  {
    "id": "paper-107",
    "title": "Synthetic Returns for Long-Term Credit Assignment",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2102.12425",
    "stars": 0
  },
  {
    "id": "paper-108",
    "title": "Been There, Done That: Meta-Learning with Episodic Recall",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Meta-Learning",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://proceedings.mlr.press/v80/ritter18a.html",
    "stars": 0
  },
  {
    "id": "paper-109",
    "title": "Reinforcement Learning, Fast and Slow",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Meta-Learning",
      "Plasticity",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(19)30061-0",
    "stars": 0
  },
  {
    "id": "paper-110",
    "title": "Rapid Task-Solving in Novel Environments",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL",
      "Skill Discovery"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2006.03662",
    "stars": 0
  },
  {
    "id": "paper-111",
    "title": "Transformers without Normalization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2503.10622",
    "stars": 0
  },
  {
    "id": "paper-112",
    "title": "nGPT: Normalized Transformer with Representation Learning on the Hypersphere",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields",
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2410.01131",
    "stars": 0
  },
  {
    "id": "paper-113",
    "title": "[Duplicate] RL Overview Book",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://x.com/curiousZeedX/status/1901866877917503645",
    "stars": 0
  },
  {
    "id": "paper-114",
    "title": "Temporal Abstraction in Reinforcement Learning with the Successor Representation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2110.05740",
    "stars": 0
  },
  {
    "id": "paper-115",
    "title": "[Duplicate] A Survey of In-Context Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2502.07978",
    "stars": 0
  },
  {
    "id": "paper-116",
    "title": "Reinforcement Learning: An Overview",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2412.05265",
    "stars": 0
  },
  {
    "id": "paper-117",
    "title": "Causal Emergence 2.0: Quantifying emergent complexity",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Emergence"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      },
      {
        "id": "adam-goldstein",
        "name": "Adam Goldstein",
        "avatar": "AG",
        "email": "adam-goldstein@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2503.13395",
    "stars": 0
  },
  {
    "id": "paper-118",
    "title": "Towards Safe and Honest AI Agents with Neural Self-Other Overlap",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2412.16325 ",
    "stars": 0
  },
  {
    "id": "paper-119",
    "title": "Diversity is All You Need: Learning Skills without a Reward Function",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Open Enededness",
      "RL",
      "Skill Discovery"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1802.06070",
    "stars": 0
  },
  {
    "id": "paper-120",
    "title": "Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2310.02782",
    "stars": 0
  },
  {
    "id": "paper-121",
    "title": "Eidetic Learning: an Efficient and Provable Solution to Catastrophic Forgetting",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2502.09500",
    "stars": 0
  },
  {
    "id": "paper-122",
    "title": "Predicting Grokking",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://openreview.net/pdf/d2213cf66f10f0572dcf98c18a7986a7e8a2a87b.pdf",
    "stars": 0
  },
  {
    "id": "paper-123",
    "title": "Finding General Equilibria in Many-Agent Economic Simulations using Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Multi-Agency",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://openreview.net/forum?id=d5IQ3k7ed__",
    "stars": 0
  },
  {
    "id": "paper-124",
    "title": "Measuring Policy Distance for Multi-Agent Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "RL"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/html/2401.11257v2",
    "stars": 0
  },
  {
    "id": "paper-125",
    "title": "Robust Autonomy Emerges from Self-Play",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2502.03349",
    "stars": 0
  },
  {
    "id": "paper-126",
    "title": "Reinforcement Learning for Long-Horizon Interactive LLM Agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "LLMs",
      "RL",
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2502.01600",
    "stars": 0
  },
  {
    "id": "paper-127",
    "title": "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "Memory",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2402.19427",
    "stars": 0
  },
  {
    "id": "paper-128",
    "title": "The Role of Fibration Symmetries in Geometric Deep Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation",
      "Evolution",
      "Exploration"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2408.15894",
    "stars": 0
  },
  {
    "id": "paper-129",
    "title": "Symmetry of Living Systems: Symmetry Fibrations and Synchronization in Biological Networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "ALife",
      "Evolution",
      "Thermodynamics"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "adam-goldstein",
        "name": "Adam Goldstein",
        "avatar": "AG",
        "email": "adam-goldstein@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2502.18713",
    "stars": 0
  },
  {
    "id": "paper-130",
    "title": "Artificial Kuramoto Oscillatory Neurons",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields",
      "Memory",
      "Neural Architecture Search",
      "Transformer"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2410.13821",
    "stars": 0
  },
  {
    "id": "paper-131",
    "title": "Muesli: Combining Improvements in Policy Optimization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2104.06159",
    "stars": 0
  },
  {
    "id": "paper-132",
    "title": "Stabilizing Transformers for Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "RL",
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1910.06764",
    "stars": 0
  },
  {
    "id": "paper-133",
    "title": "Random Latent Exploration for Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2407.13755",
    "stars": 0
  },
  {
    "id": "paper-134",
    "title": "Automating Curriculum Learning for Reinforcement Learning using a Skill-Based Bayesian Network",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/2502.15662",
    "stars": 0
  },
  {
    "id": "paper-135",
    "title": "Evaluation Mechanism of Collective Intelligence for Heterogeneous Agents Group",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/1903.00206",
    "stars": 0
  },
  {
    "id": "paper-136",
    "title": "Measuring collaborative emergent behaviour in multi-agent reinforcement learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/1807.08663",
    "stars": 0
  },
  {
    "id": "paper-137",
    "title": "Boundaries Sequence",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://www.alignmentforum.org/s/LWJsgNYE8wzv49yEc",
    "stars": 0
  },
  {
    "id": "paper-138",
    "title": "[Duplicate] Self-Other Overlap: A Neglected Approach to AI Alignment",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment",
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://www.lesswrong.com/posts/hzt9gHpNwA2oHtwKX/self-other-overlap-a-neglected-approach-to-ai-alignment",
    "stars": 0
  },
  {
    "id": "paper-139",
    "title": "Multi-agent cooperation through learning-aware policy gradients",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2410.18636",
    "stars": 0
  },
  {
    "id": "paper-140",
    "title": "Relational Norms for Human-AI Cooperation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Human-AI interaction"
    ],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-141",
    "title": "Deep Hierarchical Planning from Pixels",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL",
      "Trajectory Planning"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2206.04114",
    "stars": 0
  },
  {
    "id": "paper-142",
    "title": "A Survey of In-Context Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2502.07978",
    "stars": 0
  },
  {
    "id": "paper-143",
    "title": "Reevaluating Policy Gradient Methods for Imperfect-Information Games",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://www.arxiv.org/abs/2502.08938",
    "stars": 0
  },
  {
    "id": "paper-144",
    "title": "Hyperbolic Brain Representations",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "ALife",
      "Hyperbolic geometry / Conformal Fields",
      "Neural Architecture Search"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2409.12990v1",
    "stars": 0
  },
  {
    "id": "paper-145",
    "title": "Cultural Evolution of Cooperation among LLM Agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2412.10270",
    "stars": 0
  },
  {
    "id": "paper-146",
    "title": "Agency is frame-dependent",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Agency and alignment"
    ],
    "readBy": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2502.04403",
    "stars": 0
  },
  {
    "id": "paper-147",
    "title": "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "Neural Architecture Search",
      "RL",
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.arxiv.org/abs/2502.05171",
    "stars": 0
  },
  {
    "id": "paper-148",
    "title": "AMAGO-2: Breaking the Multi-Task Barrier in Meta-Reinforcement Learning with Transformers",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2411.11188",
    "stars": 0
  },
  {
    "id": "paper-149",
    "title": "Towards General-Purpose Model-Free Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2501.16142",
    "stars": 0
  },
  {
    "id": "paper-150",
    "title": "Selective Sequence Modeling",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-151",
    "title": "Decision Mamba: Reinforcement Learning via Hybrid Selective Sequence Modeling",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "LLMs",
      "Meta-Learning",
      "RL",
      "Transformer",
      "Trajectory Planning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://openreview.net/pdf?id=wFzIMbTsY7",
    "stars": 0
  },
  {
    "id": "paper-152",
    "title": "Mike Levin paper: Ingressing Minds: Causal Patterns Beyond Genetics and Environment in Natural, Synthetic, and Hybrid Embodiments",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "https://osf.io/preprints/psyarxiv/5g2xj_v1",
    "stars": 0
  },
  {
    "id": "paper-153",
    "title": "Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2206.15378",
    "stars": 0
  },
  {
    "id": "paper-154",
    "title": "Learning to (Learn at Test Time): RNNs with Expressive Hidden States",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "Memory",
      "Meta-Learning",
      "Neural Architecture Search",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2407.04620",
    "stars": 3
  },
  {
    "id": "paper-155",
    "title": "Towards a theory of learning dynamics in deep state space models",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "Memory"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2407.07279",
    "stars": 0
  },
  {
    "id": "paper-156",
    "title": "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "LLMs",
      "Neural Architecture Search",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2405.21060",
    "stars": 0
  },
  {
    "id": "paper-157",
    "title": "Minimax-01 hybrid lightning attention",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "LLMs"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2501.08313",
    "stars": 0
  },
  {
    "id": "paper-158",
    "title": "MiniMax-01: Scaling Foundation Models with Lightning Attention",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "LLMs",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2501.08313",
    "stars": 0
  },
  {
    "id": "paper-159",
    "title": "Modern Sequence Models in Context of Multi-Agent Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "LLMs",
      "Multi-Agency",
      "RL",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://epub.jku.at/obvulihs/download/pdf/10580112",
    "stars": 0
  },
  {
    "id": "paper-160",
    "title": "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "LLMs",
      "Meta-Learning",
      "RL",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2404.05892",
    "stars": 0
  },
  {
    "id": "paper-161",
    "title": "STaR: Bootstrapping Reasoning With Reasoning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "LLMs",
      "RL"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2203.14465",
    "stars": 0
  },
  {
    "id": "paper-162",
    "title": "The Factory Must Grow: Automation in Factorio",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Open Enededness",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2102.04871",
    "stars": 0
  },
  {
    "id": "paper-163",
    "title": "A free energy principle for generic quantum systems",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "adam-goldstein",
        "name": "Adam Goldstein",
        "avatar": "AG",
        "email": "adam-goldstein@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2112.15242",
    "stars": 3
  },
  {
    "id": "paper-164",
    "title": "A mosaic of Chu Spaces and Channel Theory",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://www.tandfonline.com/doi/full/10.1080/0952813X.2018.1544282",
    "stars": 1
  },
  {
    "id": "paper-165",
    "title": "Maximum Entropy On-Policy Actor-Critic via Entropy Advantage Estimation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2407.18143",
    "stars": 0
  },
  {
    "id": "paper-166",
    "title": "DeepSeek: Pushing the Limits of Mathematical Reasoning in Open Language Models",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2402.03300",
    "stars": 0
  },
  {
    "id": "paper-167",
    "title": "DeepSeekMath: Pushing the Limits of Mathematical",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "LLMs",
      "Open Enededness",
      "RL"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2402.03300",
    "stars": 0
  },
  {
    "id": "paper-168",
    "title": "Evolution and the Knightian Blindspot of Machine Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Open Enededness"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2501.13075",
    "stars": 0
  },
  {
    "id": "paper-169",
    "title": "Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2010.14498",
    "stars": 2
  },
  {
    "id": "paper-170",
    "title": "The Fifth Corner of Four - Priest 2018",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "adam-goldstein",
        "name": "Adam Goldstein",
        "avatar": "AG",
        "email": "adam-goldstein@softmax.ai"
      }
    ],
    "link": "",
    "stars": 3
  },
  {
    "id": "paper-171",
    "title": "Madhyamaka, Ultimate Reality, and Ineffability",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 2
  },
  {
    "id": "paper-172",
    "title": "Metatheory and dialethiesm - Priest 2020",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 2
  },
  {
    "id": "paper-173",
    "title": "Transcending the Ultimate Duality - Priest 2023",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 3
  },
  {
    "id": "paper-174",
    "title": "Thoughts and thinkers: On the complementarity between objects and processes - Fields & Levin 2025",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 3
  },
  {
    "id": "paper-175",
    "title": "Science Generates Limit Paradoxes - Dietrich & Fields 2015",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 3
  },
  {
    "id": "paper-176",
    "title": "Nagarjuna and the Limits of Thought - Garfield & Priest 2003",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 3
  },
  {
    "id": "paper-177",
    "title": "Category Theory and the Ontology of Śūnyata",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meditation, Insight, Emptiness"
    ],
    "readBy": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "",
    "stars": 2
  },
  {
    "id": "paper-178",
    "title": "Heuristically Adaptive Diffusion-Model Evolutionary Strategy",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2411.13420",
    "stars": 0
  },
  {
    "id": "paper-179",
    "title": "A Comprehensive Survey of Continual Learning: Theory, Method and Application",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Continual Learning",
      "Plasticity",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2302.00487",
    "stars": 0
  },
  {
    "id": "paper-180",
    "title": "Summarizing societies: Agent abstraction in multi-agent reinforcement learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Avse5gIAAAAJ&sortby=pubdate&citation_for_view=Avse5gIAAAAJ:jU7OWUQzBzMC",
    "stars": 0
  },
  {
    "id": "paper-181",
    "title": "On Bayesian Mechanics: A Physics of and by Beliefs",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Bayesian Mechanics"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2205.11543",
    "stars": 0
  },
  {
    "id": "paper-182",
    "title": "VIME: Variational Information Maximizing Exploration",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1605.09674",
    "stars": 1
  },
  {
    "id": "paper-183",
    "title": "Generative Teaching Networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "Neural Architecture Search",
      "Unsupervised Environment Design"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/1912.07768",
    "stars": 1
  },
  {
    "id": "paper-184",
    "title": "Loss of plasticity in deep continual learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "matthew-bull",
        "name": "Matthew Bull",
        "avatar": "MB",
        "email": "matthew-bull@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://www.nature.com/articles/s41586-024-07711-7",
    "stars": 0
  },
  {
    "id": "paper-185",
    "title": "Maintaining Plasticity in Continual Learning via Regenerative Regularization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/html/2308.11958v3",
    "stars": 0
  },
  {
    "id": "paper-186",
    "title": "Deep RL with plasticity injection",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://proceedings.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf",
    "stars": 0
  },
  {
    "id": "paper-187",
    "title": "Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [
      {
        "id": "matthew-bull",
        "name": "Matthew Bull",
        "avatar": "MB",
        "email": "matthew-bull@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2407.01704",
    "stars": 0
  },
  {
    "id": "paper-188",
    "title": "Weight Clipping for Deep Continual and Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2407.01704",
    "stars": 0
  },
  {
    "id": "paper-189",
    "title": "Understanding Plasticity in Neural Networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/2303.01486",
    "stars": 0
  },
  {
    "id": "paper-190",
    "title": "Titans: Learning to Memorize at Test Time",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Meta-Learning"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2501.00663",
    "stars": 3
  },
  {
    "id": "paper-191",
    "title": "Techniques for training large neural networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Optimization"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://openai.com/index/techniques-for-training-large-neural-networks/",
    "stars": 0
  },
  {
    "id": "paper-192",
    "title": "Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://papers.nips.cc/paper_files/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html",
    "stars": 0
  },
  {
    "id": "paper-193",
    "title": "Distributed Deep Reinforcement Learning: An Overview",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2011.11012",
    "stars": 0
  },
  {
    "id": "paper-194",
    "title": "Reinforcement Learning with Information-Theoretic Actuation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Skill Discovery",
      "Trajectory Planning"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2109.15147",
    "stars": 1
  },
  {
    "id": "paper-195",
    "title": "Human-level play in the game of Diplomacy by combining language models with strategic reasoning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://noambrown.github.io/papers/22-Science-Diplomacy-TR.pdf",
    "stars": 0
  },
  {
    "id": "paper-196",
    "title": "RL2: Fast Reinforcement Learning via Slow Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1611.02779",
    "stars": 0
  },
  {
    "id": "paper-197",
    "title": "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Open Enededness",
      "RL",
      "Skill Discovery",
      "Unsupervised Environment Design"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2402.16801",
    "stars": 2
  },
  {
    "id": "paper-198",
    "title": "Open-Ended Learning Leads to Generally Capable Agents (XLand)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Open Enededness",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2107.12808",
    "stars": 3
  },
  {
    "id": "paper-199",
    "title": "AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Open Enededness",
      "Unsupervised Environment Design"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1905.10985",
    "stars": 3
  },
  {
    "id": "paper-200",
    "title": "Quantum tensor product structures are observable-induced",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/quant-ph/0308043",
    "stars": 0
  },
  {
    "id": "paper-201",
    "title": "Reinforcement Learning with Unsupervised Auxiliary Tasks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1611.05397",
    "stars": 0
  },
  {
    "id": "paper-202",
    "title": "Mix&Match - Agent Curricula for Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Multi-Agency",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1806.01780",
    "stars": 0
  },
  {
    "id": "paper-203",
    "title": "Local minima in training of neural networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Optimization"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1611.06310",
    "stars": 0
  },
  {
    "id": "paper-204",
    "title": "Progress & Compress: A scalable framework for continual learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1805.06370",
    "stars": 0
  },
  {
    "id": "paper-205",
    "title": "Evolving intrinsic motivations for altruistic behavior",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation",
      "Evolution",
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1811.05931",
    "stars": 0
  },
  {
    "id": "paper-206",
    "title": "Distilling Policy Distillation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1902.02186",
    "stars": 0
  },
  {
    "id": "paper-207",
    "title": "$α$-Rank: Multi-Agent Evaluation by Evolution",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution",
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1903.01373",
    "stars": 0
  },
  {
    "id": "paper-208",
    "title": "Smooth markets: A basic mechanism for organizing gradient-based learners",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Optimization"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2001.04678",
    "stars": 0
  },
  {
    "id": "paper-209",
    "title": "A Limited-Capacity Minimax Theorem for Non-Convex Games or: How I Learned to Stop Worrying about Mixed-Nash and Love Neural Nets",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2002.05820",
    "stars": 0
  },
  {
    "id": "paper-210",
    "title": "Perception-Prediction-Reaction Agents for Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2006.15223",
    "stars": 0
  },
  {
    "id": "paper-211",
    "title": "Negotiating Team Formation Using Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation",
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2010.10380",
    "stars": 0
  },
  {
    "id": "paper-212",
    "title": "Pick Your Battles: Interaction Graphs as Population-Level Objectives for Strategic Diversity",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2110.04041",
    "stars": 0
  },
  {
    "id": "paper-213",
    "title": "On the Limitations of Elo: Real-World Games, are Transitive, not Additive",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2206.12301",
    "stars": 0
  },
  {
    "id": "paper-214",
    "title": "Unicorn: Continual Learning with a Universal, Off-policy Agent",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation",
      "Plasticity"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1802.08294",
    "stars": 0
  },
  {
    "id": "paper-215",
    "title": "Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Exploration",
      "Open Enededness",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2305.13795",
    "stars": 0
  },
  {
    "id": "paper-216",
    "title": "OFENet: Can Increasing Input Dimensionality Improve Deep Reinforcement Learning?",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2003.01629",
    "stars": 2
  },
  {
    "id": "paper-217",
    "title": "Exploring Zero-Shot Emergent Communication in Embodied Multi-Agent Populations",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2010.15896",
    "stars": 0
  },
  {
    "id": "paper-218",
    "title": "Zero shot communication - three papers",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-219",
    "title": "Multi-Agent Cooperation and the Emergence of (Natural) Language",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1612.07182",
    "stars": 0
  },
  {
    "id": "paper-220",
    "title": "Autocurricula and the Emergence of Innovation from Social Interaction: A Manifesto for Multi-Agent Intelligence Research",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1903.00742",
    "stars": 0
  },
  {
    "id": "paper-221",
    "title": "A Review of Cooperation in Multi-agent Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2312.05162",
    "stars": 0
  },
  {
    "id": "paper-222",
    "title": "“Other-Play” for Zero-Shot Coordination Hengyuan",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/2003.02979",
    "stars": 0
  },
  {
    "id": "paper-223",
    "title": "Few-shot Language Coordination by Modeling Theory of Mind",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2107.05697",
    "stars": 0
  },
  {
    "id": "paper-224",
    "title": "Maintaining cooperation in complex social dilemmas using deep reinforcement learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1707.01068",
    "stars": 0
  },
  {
    "id": "paper-225",
    "title": "Inequity aversion improves cooperation in intertemporal social dilemmas",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://proceedings.neurips.cc/paper/2018/hash/7fea637fd6d02b8f0adf6f7dc36aed93-Abstract.html",
    "stars": 0
  },
  {
    "id": "paper-226",
    "title": "Learning Reciprocity in Complex Sequential Social Dilemmas",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Communication & Cooperation"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1903.08082",
    "stars": 0
  },
  {
    "id": "paper-227",
    "title": "Open-Endedness is Essential for Artificial Superhuman Intelligence",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Open Enededness",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2406.04268",
    "stars": 0
  },
  {
    "id": "paper-228",
    "title": "[Duplicate] Generally capable agents emerge from open-ended play (xland)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "RL",
      "Unsupervised Environment Design"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://deepmind.google/discover/blog/generally-capable-agents-emerge-from-open-ended-play/",
    "stars": 0
  },
  {
    "id": "paper-229",
    "title": "Competency in Navigating Arbitrary Spaces as an Invariant for Analyzing Cognition in Diverse Embodiments",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Optimization",
      "Thermodynamics"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://www.mdpi.com/1099-4300/24/6/819",
    "stars": 3
  },
  {
    "id": "paper-230",
    "title": "OMNI: Open-endedness via Models of human Notions of Interestingness",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "LLMs",
      "Open Enededness",
      "RL",
      "Unsupervised Environment Design"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2306.01711",
    "stars": 0
  },
  {
    "id": "paper-231",
    "title": "OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Exploration",
      "Open Enededness",
      "RL",
      "Unsupervised Environment Design"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2405.15568",
    "stars": 0
  },
  {
    "id": "paper-232",
    "title": "Beyond the Matrix: Using Multi-Agent-Reinforcement Learning and Behavioral Experiments to Study Social-Ecological Systems",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency"
    ],
    "readBy": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://osf.io/preprints/osf/6fw42  https://docs.google.com/document/d/14Uo7uGaXoNvfJ4RwB5gcr5lzixhOadAjuv6BJxkGv8U/edit?tab=t.0",
    "stars": 0
  },
  {
    "id": "paper-233",
    "title": "Mastering Diverse Domains through World Models (Dreamer V3)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL",
      "World Model"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "Mastering Diverse Domains through World Models: https://arxiv.org/abs/2301.04104",
    "stars": 3
  },
  {
    "id": "paper-234",
    "title": "Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Evolution",
      "Exploration",
      "Open Enededness",
      "Unsupervised Environment Design"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2003.08536",
    "stars": 3
  },
  {
    "id": "paper-235",
    "title": "Exploration Unbound",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2407.12178",
    "stars": 1
  },
  {
    "id": "paper-236",
    "title": "RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://github.com/RLE-Foundation/RLeXplore",
    "stars": 0
  },
  {
    "id": "paper-237",
    "title": "Exploration via Elliptical Episodic Bonuses (E3B)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2210.05805",
    "stars": 2
  },
  {
    "id": "paper-238",
    "title": "SAMBA Safe Model-Based & Active Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2006.09436",
    "stars": 0
  },
  {
    "id": "paper-239",
    "title": "Decoupling Representation Learning from Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2009.08319",
    "stars": 0
  },
  {
    "id": "paper-240",
    "title": "ELDEN Exploration via Local Dependencies",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Multi-Agency",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2310.08702.pdf",
    "stars": 0
  },
  {
    "id": "paper-241",
    "title": "CAT-SAC Soft Actor-Critic with Curiosity-Aware Entropy Temperature",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      }
    ],
    "link": "https://openreview.net/forum?id=paE8yL0aKHo",
    "stars": 2
  },
  {
    "id": "paper-242",
    "title": "States as goal-directed concepts an epistemic approach to state-representation learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "Phenotyping"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2312.02367",
    "stars": 0
  },
  {
    "id": "paper-243",
    "title": "Active Sensing with Predictive Coding and Uncertainty Minimization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Exploration"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2307.00668",
    "stars": 0
  },
  {
    "id": "paper-244",
    "title": "Seeking entropy complex behavior from intrinsic motivation to occupy action-state path space",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2205.10316",
    "stars": 0
  },
  {
    "id": "paper-245",
    "title": "Empowerment, Free Energy Principle and Maximum Occupancy Principle Compared",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://openreview.net/forum?id=OcHrsQox0Z",
    "stars": 0
  },
  {
    "id": "paper-246",
    "title": "General Intelligence Requires Rethinking Exploration",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2211.07819",
    "stars": 2
  },
  {
    "id": "paper-247",
    "title": "[Duplicate] The Diffusion Actor-Critic with Entropy Regulator (DACER)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2405.15177",
    "stars": 0
  },
  {
    "id": "paper-248",
    "title": "Making the Thermodynamic Cost of Active Inference Explicit",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Thermodynamics"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://pubmed.ncbi.nlm.nih.gov/39202092/",
    "stars": 0
  },
  {
    "id": "paper-249",
    "title": "The free-energy principle: a unified brain theory?",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Active Inference",
      "Optimization"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf",
    "stars": 0
  },
  {
    "id": "paper-250",
    "title": "Beyond the matrix: Experimental approaches to studying cognitive agents in social-ecological systems",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Phenotyping"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://www.sciencedirect.com/science/article/abs/pii/S0010027724002798?via%3Dihub",
    "stars": 0
  },
  {
    "id": "paper-251",
    "title": "Automating the Search for Artificial Life with Foundation Models",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "ALife"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      },
      {
        "id": "lars-sandved-smith",
        "name": "Lars Sandved-Smith",
        "avatar": "LS",
        "email": "lars-sandved-smith@softmax.ai"
      },
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://sakana.ai/asal/",
    "stars": 0
  },
  {
    "id": "paper-252",
    "title": "Learning to Continually Learn (ANML)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Meta-Learning",
      "Plasticity"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2002.09571",
    "stars": 2
  },
  {
    "id": "paper-253",
    "title": "Differentiable plasticity: training plastic neural networks with backpropagation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Meta-Learning",
      "Plasticity"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1804.02464",
    "stars": 3
  },
  {
    "id": "paper-254",
    "title": "DeepSeek-V3 (highly efficient transformer training)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "Transformer"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf",
    "stars": 0
  },
  {
    "id": "paper-255",
    "title": "Hyperbolic VAE via Latent Gaussian Distributions",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2209.15217",
    "stars": 0
  },
  {
    "id": "paper-256",
    "title": "Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields",
      "Topology"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2104.02228",
    "stars": 0
  },
  {
    "id": "paper-257",
    "title": "Unsupervised Hyperbolic Representation Learning via Message Passing Auto-Encoders",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields",
      "Topology"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://openaccess.thecvf.com/content/CVPR2021/papers/Park_Unsupervised_Hyperbolic_Representation_Learning_via_Message_Passing_Auto-Encoders_CVPR_2021_paper.pdf",
    "stars": 0
  },
  {
    "id": "paper-258",
    "title": "Reinforcement Learning in Hyperbolic Spaces: Models and Experiments",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields",
      "RL",
      "Topology"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/html/2410.09466v1",
    "stars": 0
  },
  {
    "id": "paper-259",
    "title": "Hyperbolic Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields",
      "RL",
      "Topology"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://openreview.net/forum?id=TfBHFLgv77",
    "stars": 0
  },
  {
    "id": "paper-260",
    "title": "Hyperbolic deep RL for continuous control",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Hyperbolic geometry / Conformal Fields",
      "RL",
      "Topology"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "link": "https://openreview.net/forum?id=Mrz9PgP3sT",
    "stars": 0
  },
  {
    "id": "paper-261",
    "title": "Traces of Consciousness",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Consciousness",
      "Emergence"
    ],
    "readBy": [
      {
        "id": "adam-goldstein",
        "name": "Adam Goldstein",
        "avatar": "AG",
        "email": "adam-goldstein@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://www.preprints.org/manuscript/202410.1305/v1",
    "stars": 0
  },
  {
    "id": "paper-262",
    "title": "Attention Schema in Neural Agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "Phenotyping",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2305.17375",
    "stars": 0
  },
  {
    "id": "paper-263",
    "title": "Learning Attractor Dynamics for Generative Memory",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://openreview.net/forum?id=ByVISObOZS",
    "stars": 0
  },
  {
    "id": "paper-264",
    "title": "Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://openreview.net/forum?id=ByZgcd-_ZS",
    "stars": 0
  },
  {
    "id": "paper-265",
    "title": "Associative Long Short-Term Memory",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://proceedings.mlr.press/v48/danihelka16.pdf",
    "stars": 0
  },
  {
    "id": "paper-266",
    "title": "The Kanerva Machine A Generative Distributed Memory",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/1804.01756v1.pdf",
    "stars": 0
  },
  {
    "id": "paper-267",
    "title": "Recurrent Reinforcement Learning with Memoroids",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://openreview.net/forum?id=nA4Q983a1v",
    "stars": 0
  },
  {
    "id": "paper-268",
    "title": "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2006.05990",
    "stars": 0
  },
  {
    "id": "paper-269",
    "title": "The 37 Implementation Details of Proximal Policy Optimization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/",
    "stars": 0
  },
  {
    "id": "paper-270",
    "title": "Scaling laws for single-agent reinforcement learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2301.13442",
    "stars": 0
  },
  {
    "id": "paper-271",
    "title": "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1506.02438",
    "stars": 0
  },
  {
    "id": "paper-272",
    "title": "Proximal Policy Optimization Algorithms",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1707.06347",
    "stars": 0
  },
  {
    "id": "paper-273",
    "title": "AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "alex-smith",
        "name": "Alex Smith",
        "avatar": "AS",
        "email": "alex-smith@softmax.ai"
      }
    ],
    "link": "https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/  https://www.seas.upenn.edu/~cis520/papers/RL_for_starcraft.pdf",
    "stars": 0
  },
  {
    "id": "paper-274",
    "title": "Emergent tool use from multi-agent interaction",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://openai.com/index/emergent-tool-use/",
    "stars": 0
  },
  {
    "id": "paper-275",
    "title": "Capture the Flag: the emergence of complex cooperative agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://deepmind.google/discover/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/",
    "stars": 0
  },
  {
    "id": "paper-276",
    "title": "Compressive Transformers for Long-Range Sequence Modelling",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Transformer"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1911.05507",
    "stars": 0
  },
  {
    "id": "paper-277",
    "title": "Hybrid computing using a neural network with dynamic external memory",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://www.nature.com/articles/nature20101",
    "stars": 0
  },
  {
    "id": "paper-278",
    "title": "Scaling Instructable Agents Across Many Simulated Worlds (SIMA)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2404.10179",
    "stars": 0
  },
  {
    "id": "paper-279",
    "title": "Latent Plan Transformer for Trajectory Abstraction Planning as Latent Space Inference",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2402.04647",
    "stars": 0
  },
  {
    "id": "paper-280",
    "title": "Dota 2 with Large Scale Deep Reinforcement Learning (DOTA)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "andre-von-houck",
        "name": "Andre von Houck",
        "avatar": "AV",
        "email": "andre-von-houck@softmax.ai"
      }
    ],
    "link": "https://cdn.openai.com/dota-2.pdf",
    "stars": 1
  },
  {
    "id": "paper-281",
    "title": "Neural Network Surgery with Sets",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1912.06719",
    "stars": 0
  },
  {
    "id": "paper-282",
    "title": "Multi-Task Deep Reinforcement Learning with PopArt",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL",
      "Unsupervised Environment Design"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/pdf/1809.04474",
    "stars": 0
  },
  {
    "id": "paper-283",
    "title": "Fixup Initialization  Residual Learning Without Normalization",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1901.09321",
    "stars": 0
  },
  {
    "id": "paper-284",
    "title": "An Investigation of Model-Free Planning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "MCTS",
      "Memory",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1901.03559",
    "stars": 3
  },
  {
    "id": "paper-285",
    "title": "Thinker - Learning to Plan and Act",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "MCTS",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2307.14993",
    "stars": 3
  },
  {
    "id": "paper-286",
    "title": "Extending World Models for Multi-Agent Reinforcement Learning in MALMÖ",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://ceur-ws.org/Vol-2282/MARLO_110.pdf",
    "stars": 1
  },
  {
    "id": "paper-287",
    "title": "Mastering Atari Games with Limited Data (EfficientZero)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "MCTS",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": " https://github.com/YeWR/EfficientZero?tab=readme-ov-file",
    "stars": 2
  },
  {
    "id": "paper-288",
    "title": "Bigger, Better, Faster - Human-level Atari with human-level efficiency",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2305.19452",
    "stars": 2
  },
  {
    "id": "paper-289",
    "title": "Causality Detection for Efficient Multi-Agent Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-290",
    "title": "Adversarial Reinforcement Learning for Procedural Content Generation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning",
      "RL",
      "Unsupervised Environment Design"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/pdf/2103.04847",
    "stars": 0
  },
  {
    "id": "paper-291",
    "title": "Scaling Scaling Laws with Board Games",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2104.03113",
    "stars": 1
  },
  {
    "id": "paper-292",
    "title": "Were RNNs All We Needed?",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2410.01201",
    "stars": 2
  },
  {
    "id": "paper-293",
    "title": "Data-Efficient Reinforcement Learning with Self-Predictive Representations",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2007.05929",
    "stars": 2
  },
  {
    "id": "paper-294",
    "title": "AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "RL",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://ut-austin-rpl.github.io/amago/",
    "stars": 3
  },
  {
    "id": "paper-295",
    "title": "Human-Timescale Adaptation in an Open-Ended Task Space (ADA)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      },
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://sites.google.com/view/adaptive-agent/",
    "stars": 3
  },
  {
    "id": "paper-296",
    "title": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Transformer"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1901.02860",
    "stars": 0
  },
  {
    "id": "paper-297",
    "title": "Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://jaehyeon-son.github.io/_pages/icrl-mb-preprint.pdf",
    "stars": 0
  },
  {
    "id": "paper-298",
    "title": "Diversity Measures",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Open Enededness",
      "Phenotyping"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://en.wikipedia.org/wiki/Diversity_index#Simpson_index",
    "stars": 0
  },
  {
    "id": "paper-299",
    "title": "Meta-Learning an Evolvable Developmental Encoding",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "NCA"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2406.09020",
    "stars": 0
  },
  {
    "id": "paper-300",
    "title": "Benefits of Assistance over Reward Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-301",
    "title": "Learning Social Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-302",
    "title": "Too Many Cooks: Bayesian Inference for Coordinating Multi-Agent Collaboration",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [
      {
        "id": "george-deane",
        "name": "George Deane",
        "avatar": "GD",
        "email": "george-deane@softmax.ai"
      }
    ],
    "link": "https://openreview.net/pdf/d2213cf66f10f0572dcf98c18a7986a7e8a2a87b.pdf",
    "stars": 0
  },
  {
    "id": "paper-303",
    "title": "Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-304",
    "title": "Quantifying Differences in Reward Functions",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-305",
    "title": "Faster Algorithms for Optimal Ex-Ante Coordinated Collusive Strategies in Extensive-Form Zero-Sum Games",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-306",
    "title": "No-Regret Learning Dynamics for Extensive-Form Correlated Equilibrium",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-307",
    "title": "Multi-Agent Coordination through Signal Mediated Strategies",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-308",
    "title": "D3C: Reducing the Price of Anarchy in Multi-Agent Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-309",
    "title": "Human-Agent Cooperation in Bridge Bidding",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-310",
    "title": "Newton Optimization on Helmholtz Decomposition for Continuous Games",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-311",
    "title": "Learning to Design Fair and Private Voting Rules",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-312",
    "title": "Competing AI: How Competition Feedback Affects Machine Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-313",
    "title": "Interactive Inverse Reinforcement Learning for Cooperative Games",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-314",
    "title": "Learning to Solve Complex Tasks by Growing Knowledge Culturally Across Generations",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-315",
    "title": "On the Approximation of Cooperative Heterogeneous Multi-Agent Reinforcement Learning (MARL) Using Mean Field Control (MFC)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-316",
    "title": "Public Information Representation for Adversarial Team Games",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-317",
    "title": "A Fine-Tuning Approach to Belief State Modeling",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-318",
    "title": "A Taxonomy of Strategic Human Interactions in Traffic Conflicts",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-319",
    "title": "Ambiguity Can Compensate for Semantic Differences in Human-AI Communication",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-320",
    "title": "Automated Configuration and Usage of Strategy Portfolios for Bargaining",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-321",
    "title": "Bayesian Inference for Human-Robot Coordination in Parallel Play",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-322",
    "title": "Causal Multi-Agent Reinforcement Learning: Review and Open Problems",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-323",
    "title": "Coalitional Bargaining via Reinforcement Learning: An Application to Collaborative Vehicle Routing",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-324",
    "title": "Coordinated Reinforcement Learning for Optimizing Mobile Networks",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-325",
    "title": "Disinformation, Stochastic Harm, and Costly Effort: A Principal-Agent Analysis of Regulating Social Media Platforms",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [],
    "queued": [],
    "link": "",
    "stars": 0
  },
  {
    "id": "paper-326",
    "title": "Meta-Neural Cellular Automata",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "NCA"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://meetbarot.com/alife_poster.pdf",
    "stars": 0
  },
  {
    "id": "paper-327",
    "title": "Towards Self-Assembling Artificial Neural Networks through Neural Developmental Programs",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Cell",
      "NCA"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2307.08197",
    "stars": 0
  },
  {
    "id": "paper-328",
    "title": "HyperNCA: Growing Developmental Networks with Neural Cellular Automata",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "NCA"
    ],
    "readBy": [],
    "queued": [],
    "link": "http://arxiv.org/abs/2204.11674",
    "stars": 0
  },
  {
    "id": "paper-329",
    "title": "[Duplicate] HyperNCA: Growing Developmental Networks with Neural Cellular Automata",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "NCA",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2204.11674",
    "stars": 0
  },
  {
    "id": "paper-330",
    "title": "Autoverse An Evolvable Game Language for Learning Robust Embodied Agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2407.04221",
    "stars": 0
  },
  {
    "id": "paper-331",
    "title": "Optimizing Automatic Differentiation with Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Optimization",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2406.05027",
    "stars": 0
  },
  {
    "id": "paper-332",
    "title": "Training Large Language Models to Reason in a Continuous Latent Space",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Optimization",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2412.06769",
    "stars": 0
  },
  {
    "id": "paper-333",
    "title": "CMA Evolutionary Strategy (CMA-ES)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Optimization"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/1604.00772",
    "stars": 0
  },
  {
    "id": "paper-334",
    "title": "Recurrent Independent Mechanisms",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Optimization",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/1909.10893",
    "stars": 0
  },
  {
    "id": "paper-335",
    "title": "Federated Natural Policy Gradient and Actor Critic Methods for Multi-task Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2311.00201",
    "stars": 0
  },
  {
    "id": "paper-336",
    "title": "Artificial Generational Intelligence Cultural Accumulation in Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://openreview.net/forum?id=pMaCRgu8GV&noteId=R14gn22cbe",
    "stars": 0
  },
  {
    "id": "paper-337",
    "title": "Reciprocal Reward Influence Encourages Cooperation From Self-Interested Agents",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2406.01641#:~:text=This%20approach%20seeks%20to%20modify,a%20model%20of%20their%20policy.",
    "stars": 0
  },
  {
    "id": "paper-338",
    "title": "Multi-Reward Best Policy Identification",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://openreview.net/forum?id=x69O84Df2G",
    "stars": 0
  },
  {
    "id": "paper-339",
    "title": "Beyond Optimism Exploration With Partially Observable Rewards",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Exploration",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2406.13909",
    "stars": 0
  },
  {
    "id": "paper-340",
    "title": "Maximum Entropy Reinforcement Learning via Energy-Based Normalizing Flow",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2405.13629",
    "stars": 0
  },
  {
    "id": "paper-341",
    "title": "DynaMITE-RL A Dynamic Model for Improved Temporal Meta-Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2402.15957",
    "stars": 0
  },
  {
    "id": "paper-342",
    "title": "In-context Reinforcement Learning with Algorithm Distillation",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2210.14215",
    "stars": 0
  },
  {
    "id": "paper-343",
    "title": "Replay Buffer with Local Forgetting for Adapting to Local Environment Changes in Deep Model-Based Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2303.08690",
    "stars": 0
  },
  {
    "id": "paper-344",
    "title": "Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Plasticity",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2411.04034",
    "stars": 0
  },
  {
    "id": "paper-345",
    "title": "No Representation, No Trust Connecting Representation, Collapse, and Trust Issues in PPO",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2405.00662",
    "stars": 0
  },
  {
    "id": "paper-346",
    "title": "Exploring the Promise and Limits of Real-Time Recurrent Learning (RTRL)",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "Optimization",
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2305.19044",
    "stars": 0
  },
  {
    "id": "paper-347",
    "title": "Adam on Local Time Addressing Nonstationarity in RL with Relative Adam Timesteps",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Plasticity",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://openreview.net/forum?id=biAqUbAuG7",
    "stars": 0
  },
  {
    "id": "paper-348",
    "title": "The Road Less Scheduled",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2405.15682",
    "stars": 0
  },
  {
    "id": "paper-349",
    "title": "Real-Time Recurrent Learning using Trace Units in Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Memory",
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      },
      {
        "id": "matthew-bull",
        "name": "Matthew Bull",
        "avatar": "MB",
        "email": "matthew-bull@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2409.01449",
    "stars": 0
  },
  {
    "id": "paper-350",
    "title": "Diffusion Actor-Critic with Entropy Regulator",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2405.15177",
    "stars": 0
  },
  {
    "id": "paper-351",
    "title": "Latent Plan Transformer for Trajectory Abstraction Planning as Latent Space Inference]]",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Trajectory Planning"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2402.04647",
    "stars": 0
  },
  {
    "id": "paper-352",
    "title": "Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2407.07333",
    "stars": 0
  },
  {
    "id": "paper-353",
    "title": "Disentangled Unsupervised Skill Discovery for Efficient Hierarchical Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL",
      "Skill Discovery"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2410.11251",
    "stars": 0
  },
  {
    "id": "paper-354",
    "title": "No Regrets - Investigating and Improving Regret Approximations for Curriculum Discovery",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Curriculum Learning"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2408.15099",
    "stars": 1
  },
  {
    "id": "paper-355",
    "title": "Skill-aware Mutual Information Optimisation for Generalisation in Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2406.04815",
    "stars": 0
  },
  {
    "id": "paper-356",
    "title": "General-Purpose In-Context Learning by Meta-Learning Transformers",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning",
      "RL",
      "Transformer"
    ],
    "readBy": [],
    "queued": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/2212.04458",
    "stars": 0
  },
  {
    "id": "paper-357",
    "title": "Disentangled Unsupervised Skill Discovery for Efficient Hierarchical Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://jiahenghu.github.io/DUSDi-site/",
    "stars": 0
  },
  {
    "id": "paper-358",
    "title": "[Duplicate] OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Meta-Learning"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2405.15568",
    "stars": 0
  },
  {
    "id": "paper-359",
    "title": "MAESTRO: Open-Ended Environment Design for Multi-Agent Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "RL"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2303.03376",
    "stars": 0
  },
  {
    "id": "paper-360",
    "title": "InfiniteKitchen: Cross-environment Cooperation for Zero-shot Multi-agent Coordination",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Multi-Agency"
    ],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://openreview.net/forum?id=q9krBJHzVS&noteId=Tdk57iao1B",
    "stars": 0
  },
  {
    "id": "paper-361",
    "title": "Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [],
    "readBy": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/pdf/2403.02648",
    "stars": 0
  },
  {
    "id": "paper-362",
    "title": "Kickstarting Deep Reinforcement Learning",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Distillation",
      "RL"
    ],
    "readBy": [
      {
        "id": "david-bloomin",
        "name": "David Bloomin",
        "avatar": "DB",
        "email": "david-bloomin@softmax.ai"
      },
      {
        "id": "alexandros-vardakostas",
        "name": "Alexandros Vardakostas",
        "avatar": "AV",
        "email": "alexandros-vardakostas@softmax.ai"
      }
    ],
    "queued": [
      {
        "id": "daphne-demekas",
        "name": "Daphne Demekas",
        "avatar": "DD",
        "email": "daphne-demekas@softmax.ai"
      }
    ],
    "link": "https://arxiv.org/abs/1803.03835",
    "stars": 3
  },
  {
    "id": "paper-363",
    "title": "RWKV: Reinventing RNNs for the Transformer Era",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "Memory",
      "Transformer"
    ],
    "readBy": [],
    "queued": [],
    "link": "https://arxiv.org/abs/2305.13048",
    "stars": 0
  },
  {
    "id": "paper-364",
    "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
    "starred": false,
    "authors": [],
    "institutions": [],
    "tags": [
      "Linear Attention",
      "LLMs",
      "Transformer"
    ],
    "readBy": [
      {
        "id": "emmett-shear",
        "name": "Emmett Shear",
        "avatar": "ES",
        "email": "emmett-shear@softmax.ai"
      }
    ],
    "queued": [],
    "link": "https://arxiv.org/abs/2312.00752",
    "stars": 0
  }
];

// Additional users found in Asana (you may want to merge these with your existing mockUsers)
export const additionalUsers: User[] = [
  {
    "id": "adam-goldstein",
    "name": "Adam Goldstein",
    "avatar": "AG",
    "email": "adam-goldstein@softmax.ai"
  },
  {
    "id": "alex-smith",
    "name": "Alex Smith",
    "avatar": "AS",
    "email": "alex-smith@softmax.ai"
  },
  {
    "id": "alexandros-vardakostas",
    "name": "Alexandros Vardakostas",
    "avatar": "AV",
    "email": "alexandros-vardakostas@softmax.ai"
  },
  {
    "id": "andre-von-houck",
    "name": "Andre von Houck",
    "avatar": "AV",
    "email": "andre-von-houck@softmax.ai"
  },
  {
    "id": "daphne-demekas",
    "name": "Daphne Demekas",
    "avatar": "DD",
    "email": "daphne-demekas@softmax.ai"
  },
  {
    "id": "david-bloomin",
    "name": "David Bloomin",
    "avatar": "DB",
    "email": "david-bloomin@softmax.ai"
  },
  {
    "id": "dominik-farr",
    "name": "Dominik Farr",
    "avatar": "DF",
    "email": "dominik-farr@softmax.ai"
  },
  {
    "id": "emmett-shear",
    "name": "Emmett Shear",
    "avatar": "ES",
    "email": "emmett-shear@softmax.ai"
  },
  {
    "id": "george-deane",
    "name": "George Deane",
    "avatar": "GD",
    "email": "george-deane@softmax.ai"
  },
  {
    "id": "jack-eicher",
    "name": "Jack Eicher",
    "avatar": "JE",
    "email": "jack-eicher@softmax.ai"
  },
  {
    "id": "jack-heart",
    "name": "Jack Heart",
    "avatar": "JH",
    "email": "jack-heart@softmax.ai"
  },
  {
    "id": "lars-sandved-smith",
    "name": "Lars Sandved-Smith",
    "avatar": "LS",
    "email": "lars-sandved-smith@softmax.ai"
  },
  {
    "id": "manuel-razo-mejia",
    "name": "Manuel Razo-Mejia",
    "avatar": "MR",
    "email": "manuel-razo-mejia@softmax.ai"
  },
  {
    "id": "matthew-bull",
    "name": "Matthew Bull",
    "avatar": "MB",
    "email": "matthew-bull@softmax.ai"
  },
  {
    "id": "robb-walters",
    "name": "Robb Walters",
    "avatar": "RW",
    "email": "robb-walters@softmax.ai"
  },
  {
    "id": "yudhister-kumar",
    "name": "Yudhister Kumar",
    "avatar": "YK",
    "email": "yudhister-kumar@softmax.ai"
  }
];

// All unique categories/tags found in the papers
export const allCategories: string[] = [
  "ALife",
  "Active Inference",
  "Agency and alignment",
  "Bayesian Mechanics",
  "Communication & Cooperation",
  "Consciousness",
  "Continual Learning",
  "Contrastive RL",
  "Curriculum Learning",
  "Death / Rebirth",
  "Distillation",
  "Emergence",
  "Eval",
  "Evolution",
  "Exploration",
  "Goal Conditioned RL",
  "Human-AI interaction",
  "Hyperbolic geometry / Conformal Fields",
  "LLMs",
  "Linear Attention",
  "MCTS",
  "Meditation, Insight, Emptiness",
  "Memory",
  "Meta-Learning",
  "Multi-Agency",
  "Multi-Cell",
  "NCA",
  "Neural Architecture Search",
  "Open Enededness",
  "Optimization",
  "Phenotyping",
  "Plasticity",
  "RL",
  "Robust Agents",
  "Skill Discovery",
  "Thermodynamics",
  "Topology",
  "Trajectory Planning",
  "Transformer",
  "Unsupervised Environment Design",
  "World Model"
];
