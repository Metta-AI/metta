CREATE TABLE "paper" (
	"id" text PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"abstract" text,
	"authors" text[],
	"institutions" text[],
	"tags" text[],
	"link" text,
	"source" text,
	"externalId" text,
	"stars" integer DEFAULT 0,
	"starred" boolean DEFAULT false,
	"createdAt" timestamp DEFAULT now() NOT NULL,
	"updatedAt" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "user_paper_interaction" (
	"userId" text NOT NULL,
	"paperId" text NOT NULL,
	"starred" boolean DEFAULT false,
	"readAt" timestamp,
	"queued" boolean DEFAULT false,
	"notes" text
);
--> statement-breakpoint
ALTER TABLE "user_paper_interaction" ADD CONSTRAINT "user_paper_interaction_userId_user_id_fk" FOREIGN KEY ("userId") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_paper_interaction" ADD CONSTRAINT "user_paper_interaction_paperId_paper_id_fk" FOREIGN KEY ("paperId") REFERENCES "public"."paper"("id") ON DELETE no action ON UPDATE no action;