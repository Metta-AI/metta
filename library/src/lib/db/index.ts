import { drizzle } from "drizzle-orm/node-postgres";

import * as authSchema from "./schema/auth";
import * as postSchema from "./schema/post";

export const db = drizzle(process.env.DATABASE_URL, {
  schema: { ...authSchema, ...postSchema },
});
