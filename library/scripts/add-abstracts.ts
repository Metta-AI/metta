import { PrismaClient } from '@prisma/client';

// Initialize Prisma client
const prisma = new PrismaClient();

async function addAbstracts() {
  try {
    console.log('Adding abstracts to papers...');

    // Add abstracts to the papers
    await prisma.paper.update({
      where: { id: 'paper-1' },
      data: {
        abstract: `We present a novel approach to high-dimensional Bayesian optimization that leverages principal component analysis to reduce dimensionality while preserving the most informative features. Our method demonstrates significant improvements in sample efficiency and convergence speed compared to existing approaches. The theoretical analysis provides insights into the conditions under which dimensionality reduction preserves optimization performance.`,
        institutions: ['Stanford University', 'MIT'],
        tags: ['Optimization', 'Bayesian Methods', 'Dimensionality Reduction']
      }
    });

    await prisma.paper.update({
      where: { id: 'paper-2' },
      data: {
        abstract: `This paper introduces a novel curriculum learning framework that uses regret-based environment design to automatically generate training curricula. Our approach adaptively adjusts task difficulty based on the agent's learning progress, leading to faster convergence and better final performance. We demonstrate the effectiveness of our method on a variety of reinforcement learning benchmarks.`,
        institutions: ['UC Berkeley', 'DeepMind'],
        tags: ['Curriculum Learning', 'Reinforcement Learning', 'Environment Design']
      }
    });

    await prisma.paper.update({
      where: { id: 'paper-3' },
      data: {
        abstract: `We investigate how evolutionary algorithms can guide policy gradient methods in reinforcement learning. Our approach combines the exploration capabilities of evolution strategies with the sample efficiency of policy gradients, resulting in a hybrid method that outperforms both approaches individually. The theoretical analysis provides convergence guarantees and practical insights.`,
        institutions: ['CMU', 'Google Research'],
        tags: ['Evolutionary Algorithms', 'Policy Gradients', 'Reinforcement Learning']
      }
    });

    await prisma.paper.update({
      where: { id: 'paper-4' },
      data: {
        abstract: `This work examines the relationship between weight resampling strategies and optimizer choice in continual learning scenarios. We provide a comprehensive analysis of how different optimizers interact with weight regularization techniques, revealing insights into the mechanisms of catastrophic forgetting. Our findings suggest new approaches for mitigating forgetting in neural networks.`,
        institutions: ['University of Toronto', 'Microsoft Research'],
        tags: ['Continual Learning', 'Catastrophic Forgetting', 'Optimization']
      }
    });

    await prisma.paper.update({
      where: { id: 'paper-5' },
      data: {
        abstract: `We present an axiomatic approach to entropy and diversity measures, establishing a unified framework for understanding these fundamental concepts. Our analysis reveals deep connections between information theory and diversity quantification, leading to new measures that satisfy desirable mathematical properties. Applications to machine learning and data science are discussed.`,
        institutions: ['Princeton University', 'Facebook AI Research'],
        tags: ['Information Theory', 'Diversity Measures', 'Axiomatic Methods']
      }
    });

    console.log('Abstracts added successfully!');
  } catch (error) {
    console.error('Error adding abstracts:', error);
  } finally {
    await prisma.$disconnect();
  }
}

addAbstracts(); 