/**
 * Comment data structure for the feed with nested replies support
 */
export type CommentDTO = {
  id: string;
  content: string;
  postId: string;
  parentId: string | null;
  isBot?: boolean;
  botType?: string | null;
  author: {
    id: string;
    name: string | null;
    email: string | null;
    image: string | null;
  };
  createdAt: Date;
  updatedAt: Date;
  replies?: CommentDTO[];
  depth?: number;
};
