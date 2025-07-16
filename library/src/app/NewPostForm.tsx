"use client";
import { useAction } from "next-safe-action/hooks";
import { FC } from "react";

import { TextInput } from "@/components/TextInput";
import { createPostAction } from "@/posts/actions/createPostAction";

export const NewPostForm: FC = () => {
  const { execute } = useAction(createPostAction, {
    onSuccess: () => {
      // The feed is paginated, and patsginated state is stored on the
      // client side only.  So refreshing the entire page is the easiest way to
      // update the list of posts.
      //
      // Alternative (that would be harder to implement): pass `prepend` helper
      // from `usePaginator` to this component, and call it when the action
      // succeeds.
      window.location.reload();
    },
  });

  return (
    <form action={execute} className="flex flex-col gap-2">
      <TextInput name="title" label="Title" />
      <button
        className="rounded bg-blue-500 px-2 py-1 text-white hover:bg-blue-600"
        type="submit"
      >
        Create Post
      </button>
    </form>
  );
};
