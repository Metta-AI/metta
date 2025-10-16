"use client";

import { useRouter } from "next/navigation";
import { FC, useState, useEffect } from "react";

import { FeedPostDTO } from "@/posts/data/feed";
import { PaperCard } from "@/components/PaperCard";
import { DeleteConfirmationModal } from "@/components/DeleteConfirmationModal";
import { PhotoViewer } from "@/components/PhotoViewer";
import { SilentArxivRefresh } from "@/components/SilentArxivRefresh";
import { InPostComments } from "@/components/InPostComments";
import {
  PostHeader,
  PostActionButtons,
  AttachedImages,
  QuotedPostCard,
  PostContent,
} from "@/components/feed";
import { useDeletePost } from "@/hooks/mutations";

/**
 * FeedPost Component
 *
 * Displays a single post in the social feed with rich formatting including:
 * - Author information with avatar
 * - Post content with LaTeX support (rendered by parent component)
 * - Social metrics (likes, replies)
 * - Paper references when applicable using PaperCard
 * - Interactive elements
 */
export const FeedPost: FC<{
  post: FeedPostDTO;
  onPaperClick?: (paperId: string) => void;
  onUserClick?: (userId: string) => void;
  currentUser: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
  isCommentsExpanded: boolean;
  onCommentToggle: () => void;
  highlightedCommentId?: string | null;
  onPostDeleted?: (postId: string) => void;
}> = ({
  post,
  onPaperClick,
  onUserClick,
  currentUser,
  isCommentsExpanded,
  onCommentToggle,
  highlightedCommentId,
  onPostDeleted,
}) => {
  const router = useRouter();

  // Local state for paper data that can be updated when institutions are added
  const [paperData, setPaperData] = useState(post.paper);

  // Local state for comment count to handle immediate UI updates
  const [commentCount, setCommentCount] = useState(post.replies);

  // Delete modal state
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  // Photo viewer state
  const [isPhotoViewerOpen, setIsPhotoViewerOpen] = useState(false);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);

  // Sync local state when post prop changes
  useEffect(() => {
    setPaperData(post.paper);
    setCommentCount(post.replies);
  }, [post.paper, post.replies]);

  // Callback to update paper data when institutions are added
  const handleInstitutionsAdded = (institutions: string[]) => {
    if (paperData) {
      setPaperData({
        ...paperData,
        institutions: institutions,
      });
    }
  };

  // Handle comment count updates for immediate UI feedback
  const handleCommentCountChange = (delta: number) => {
    setCommentCount((prev) => Math.max(0, prev + delta));
  };

  // Handle quote post action
  const handleQuotePost = (e: React.MouseEvent) => {
    e.stopPropagation();

    // Check if there's already a quote draft
    const existingDraft = sessionStorage.getItem("quote-draft");
    let quotedPostIds: string[] = [post.id];
    let content = "";

    if (existingDraft) {
      try {
        const parsed = JSON.parse(existingDraft);
        quotedPostIds = parsed.quotedPostIds || [];
        content = parsed.content || "";

        // Add this post if not already included and under limit
        if (!quotedPostIds.includes(post.id) && quotedPostIds.length < 2) {
          quotedPostIds.push(post.id);
        }
      } catch (error) {
        console.error("Error parsing existing quote draft:", error);
        quotedPostIds = [post.id];
      }
    }

    // Build content with all quoted posts
    const postUrl = `${window.location.origin}/posts/${post.id}`;
    if (quotedPostIds.length === 1) {
      content = `Quoting this post:\n\n${postUrl}\n\n`;
    } else {
      // If adding a second post, append to existing content
      if (!content.includes(postUrl)) {
        content = content.trim() + `\n\n${postUrl}\n\n`;
      }
    }

    // Store the updated quote data in sessionStorage
    sessionStorage.setItem(
      "quote-draft",
      JSON.stringify({
        quotedPostIds,
        content: content,
      })
    );

    // Navigate to home page where the new post form will pick up the draft
    router.push("/?quote=" + post.id);
  };

  // Mutations
  const deletePostMutation = useDeletePost();

  // Check if current user can delete this post
  const canDelete = currentUser && currentUser.id === post.author.id;

  const handlePostClick = (e: React.MouseEvent) => {
    // Check if the click came from an image container
    const target = e.target as HTMLElement;
    const clickedImageContainer = target.closest(
      '[data-image-container="true"]'
    );

    if (clickedImageContainer) {
      // Don't handle post click if clicking on an image
      return;
    }

    // Navigate to post page instead of expanding in-place
    router.push(`/posts/${post.id}`);
  };

  const handleDelete = () => {
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = () => {
    deletePostMutation.mutate(
      { postId: post.id },
      {
        onSuccess: () => {
          // Optimistically remove post from feed
          onPostDeleted?.(post.id);
        },
      }
    );
    setShowDeleteModal(false);
  };

  const handleCancelDelete = () => {
    setShowDeleteModal(false);
  };

  // Photo viewer handlers
  const handleImageClick = (imageIndex: number) => {
    setSelectedImageIndex(imageIndex);
    setIsPhotoViewerOpen(true);
  };

  const handlePhotoViewerClose = () => {
    setIsPhotoViewerOpen(false);
  };

  const isPurePaper = post.postType === "pure-paper" && !!post.paper;

  // Photo viewer component (used in both branches)
  const photoViewer = post.images && post.images.length > 0 && (
    <PhotoViewer
      images={post.images}
      initialIndex={selectedImageIndex}
      isOpen={isPhotoViewerOpen}
      onClose={handlePhotoViewerClose}
      postAuthor={
        post.author.name || post.author.email?.split("@")[0] || "Unknown User"
      }
    />
  );

  // Delete modal component (used in both branches)
  const deleteModal = (
    <DeleteConfirmationModal
      isOpen={showDeleteModal}
      onClose={handleCancelDelete}
      onConfirm={handleConfirmDelete}
      title="Delete Post"
      message="Are you sure you want to delete this post? This action cannot be undone and will permanently remove the post and all its comments from the feed."
      isDeleting={deletePostMutation.isPending}
    />
  );

  return (
    <div
      role="button"
      onClick={handlePostClick}
      className={`group relative cursor-pointer overflow-hidden rounded-2xl shadow-sm transition before:absolute before:top-0 before:bottom-0 before:left-0 before:w-1 before:bg-transparent before:content-[''] hover:before:bg-neutral-900/70 ${
        isCommentsExpanded
          ? "border border-neutral-200 bg-white ring-2 ring-neutral-900/10"
          : "border border-neutral-200 bg-white hover:border-neutral-300 hover:bg-neutral-50"
      }`}
      style={
        isPurePaper
          ? {
              paddingBottom: "2px",
              paddingLeft: "24px",
              paddingTop: "24px",
            }
          : undefined
      }
    >
      {/* Post header with user info */}
      <div
        className={
          isPurePaper
            ? "mb-4 flex items-center justify-between"
            : "flex items-center justify-between px-4 pt-4 pb-2"
        }
      >
        <PostHeader
          author={post.author}
          createdAt={post.createdAt}
          onUserClick={onUserClick}
        />

        <PostActionButtons
          postId={post.id}
          commentCount={commentCount}
          canDelete={canDelete}
          isPurePaper={isPurePaper}
          isDeletePending={deletePostMutation.isPending}
          onDelete={handleDelete}
          onQuote={isPurePaper ? undefined : handleQuotePost}
          onCommentClick={handlePostClick}
        />
      </div>

      {/* Post content with LaTeX support (regular posts only) */}
      {!isPurePaper && post.content && <PostContent content={post.content} />}

      {/* Attached images */}
      <AttachedImages
        images={post.images || []}
        onImageClick={handleImageClick}
      />

      {/* Quoted posts display (regular posts only) */}
      {!isPurePaper && post.quotedPosts && post.quotedPosts.length > 0 && (
        <div className="px-4 pb-4">
          <div className="space-y-2">
            {post.quotedPosts.map((quotedPost) => (
              <QuotedPostCard key={quotedPost.id} quotedPost={quotedPost} />
            ))}
          </div>
        </div>
      )}

      {/* Paper card (both pure-paper and paper-post) */}
      {((isPurePaper && post.paper) ||
        (post.postType === "paper-post" && paperData)) && (
        <div className="px-4 pb-4" onClick={(e) => e.stopPropagation()}>
          <PaperCard
            paper={(isPurePaper ? post.paper : paperData)!}
            onPaperClick={onPaperClick}
          />
        </div>
      )}

      {/* Silent refresh for arXiv institution processing (regular posts only) */}
      {!isPurePaper && post.content && (
        <SilentArxivRefresh
          postId={post.id}
          content={post.content}
          onInstitutionsAdded={handleInstitutionsAdded}
        />
      )}

      {/* In-post comments */}
      <div onClick={(e) => e.stopPropagation()}>
        <InPostComments
          post={post}
          isExpanded={isCommentsExpanded}
          onToggle={onCommentToggle}
          currentUser={currentUser}
          onCommentCountChange={
            isPurePaper ? undefined : handleCommentCountChange
          }
          highlightedCommentId={highlightedCommentId}
        />
      </div>

      {deleteModal}
      {photoViewer}
    </div>
  );
};
