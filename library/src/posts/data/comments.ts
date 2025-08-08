/**
 * Comment data structure for the feed
 */
export type CommentDTO = {
  id: string;
  content: string;
  postId: string;
  author: {
    id: string;
    name: string | null;
    email: string | null;
    image: string | null;
  };
  createdAt: Date;
  updatedAt: Date;
};
