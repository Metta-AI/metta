import { db } from "../src/lib/db";
import { postsTable } from "../src/lib/db/schema/post";
import { papersTable } from "../src/lib/db/schema/paper";
import { usersTable } from "../src/lib/db/schema/auth";

async function createTestPosts() {
  try {
    // Get the first user (brandon@stem.ai)
    const users = await db.select().from(usersTable).limit(1);
    if (users.length === 0) {
      console.error('No users found in database');
      return;
    }
    const userId = users[0].id;

    // Get some papers to reference
    const papers = await db.select().from(papersTable).limit(3);
    if (papers.length === 0) {
      console.error('No papers found in database');
      return;
    }

    console.log('Creating test posts...');

    // Create a user post with LaTeX content
    await db.insert(postsTable).values({
      title: "Universal Approximation Theorem Discussion",
      content: `Fascinating discussion about the universal approximation theorem in our seminar today! For any continuous function $f: [0,1]^n \\rightarrow \\mathbb{R}$ and $\\epsilon > 0$, there exists a neural network with one hidden layer that can approximate $f$ within $\\epsilon$ error. Formally:

$$|f(x) - \\sum_{i=1}^N \\alpha_i \\sigma(w_i^T x + b_i)| < \\epsilon$$

where $\\sigma$ is a sigmoid activation function. This theoretical result explains why neural networks are so powerful! 🧠 #NeuralNetworks #Theory`,
      postType: 'user-post',
      authorId: userId,
      likes: 67,
      retweets: 23,
      replies: 12,
    });

    // Create a paper post with commentary
    await db.insert(postsTable).values({
      title: "Natural Policy Gradients Paper",
      content: `Excited to share our new paper on reinforcement learning with continuous action spaces! We introduce a novel policy gradient method that uses the natural gradient $\\nabla_\\theta J(\\theta) = F^{-1}(\\theta) \\nabla_\\theta J(\\theta)$ where $F(\\theta)$ is the Fisher information matrix. This leads to more stable training and better sample efficiency.`,
      postType: 'paper-post',
      paperId: papers[0].id,
      authorId: userId,
      likes: 42,
      retweets: 15,
      replies: 7,
    });

    // Create a pure paper post
    await db.insert(postsTable).values({
      title: "Attention Is All You Need",
      content: null,
      postType: 'pure-paper',
      paperId: papers[1].id,
      authorId: userId,
      likes: 156,
      retweets: 89,
      replies: 34,
    });

    // Create another user post with LaTeX
    await db.insert(postsTable).values({
      title: "Variational Autoencoders Implementation",
      content: `Just finished implementing variational autoencoders (VAEs)! The ELBO objective is:

$$\\mathcal{L} = \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - D_{KL}(q_\\phi(z|x) \\| p(z))$$

The first term is the reconstruction loss, and the second is the KL divergence that regularizes the latent space. The reparameterization trick $z = \\mu + \\sigma \\odot \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0,I)$ makes training possible through backpropagation! 🎨 #VAE #GenerativeModels`,
      postType: 'user-post',
      authorId: userId,
      likes: 53,
      retweets: 18,
      replies: 9,
    });

    console.log('Test posts created successfully!');
  } catch (error) {
    console.error('Error creating test posts:', error);
  }
}

createTestPosts(); 