import z from "zod/v4";

import { API_URL } from "@/server/constants";

const commonJsonMetaSchema = {
  title: z.string().optional(),
  description: z.string().optional(),
  deprecated: z.boolean().optional(),
  // injected by TS code
  isTopLevelDef: z.boolean().optional(),
};

const nullJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  type: z.literal("null"),
});

const stringJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  type: z.literal("string"),
});

const numberJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  type: z.literal("number"),
});

const integerJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  type: z.literal("integer"),
});

const booleanJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  type: z.literal("boolean"),
});

const objectJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  type: z.literal("object"),
  get additionalProperties(): z.ZodUnion<
    readonly [
      TypedJsonMetaSchema,
      typeof refJsonMetaSchema,
      typeof anyOfJsonMetaSchema,
      typeof oneOfJsonMetaSchema,
      typeof anyJsonMetaSchema,
      z.ZodLiteral<false>,
    ]
  > {
    return z.union([
      typedJsonMetaSchema,
      refJsonMetaSchema,
      anyOfJsonMetaSchema,
      oneOfJsonMetaSchema,
      anyJsonMetaSchema,
      z.literal(false),
    ]);
  },
  get properties(): z.ZodOptional<
    z.ZodRecord<
      z.ZodString,
      z.ZodUnion<
        readonly [
          TypedJsonMetaSchema,
          typeof refJsonMetaSchema,
          typeof anyOfJsonMetaSchema,
          typeof oneOfJsonMetaSchema,
          typeof anyJsonMetaSchema,
        ]
      >
    >
  > {
    return z.record(z.string(), jsonMetaSchema).optional();
  },
});

const arrayJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  type: z.literal("array"),
  get items(): z.ZodOptional<
    z.ZodUnion<
      readonly [
        TypedJsonMetaSchema,
        typeof refJsonMetaSchema,
        typeof anyOfJsonMetaSchema,
        typeof oneOfJsonMetaSchema,
        typeof anyJsonMetaSchema,
      ]
    >
  > {
    return jsonMetaSchema.optional();
  },
});

type TypedJsonMetaSchema = z.ZodDiscriminatedUnion<
  [
    typeof nullJsonMetaSchema,
    typeof stringJsonMetaSchema,
    typeof numberJsonMetaSchema,
    typeof integerJsonMetaSchema,
    typeof booleanJsonMetaSchema,
    typeof objectJsonMetaSchema,
    typeof arrayJsonMetaSchema,
  ]
>;

const typedJsonMetaSchema: TypedJsonMetaSchema = z.discriminatedUnion("type", [
  nullJsonMetaSchema,
  stringJsonMetaSchema,
  numberJsonMetaSchema,
  integerJsonMetaSchema,
  booleanJsonMetaSchema,
  objectJsonMetaSchema,
  arrayJsonMetaSchema,
]);

const anyOfJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  get anyOf(): z.ZodArray<
    z.ZodUnion<
      [TypedJsonMetaSchema, typeof refJsonMetaSchema, typeof anyJsonMetaSchema]
    >
  > {
    return z.array(
      z.union([typedJsonMetaSchema, refJsonMetaSchema, anyJsonMetaSchema])
    );
  },
});

const oneOfJsonMetaSchema = z.object({
  ...commonJsonMetaSchema,
  get oneOf(): z.ZodArray<
    z.ZodUnion<
      [TypedJsonMetaSchema, typeof refJsonMetaSchema, typeof anyJsonMetaSchema]
    >
  > {
    return z.array(
      z.union([typedJsonMetaSchema, refJsonMetaSchema, anyJsonMetaSchema])
    );
  },
  discriminator: z
    .object({
      get mapping(): z.ZodRecord<z.ZodString, z.ZodString> {
        return z.record(z.string(), z.string());
      },
      propertyName: z.string(),
    })
    .optional(),
});

export type OneOfJsonMetaSchema = z.infer<typeof oneOfJsonMetaSchema>;

const refJsonMetaSchema = z.object({
  $ref: z.string(),
});

const anyJsonMetaSchema = z.object(commonJsonMetaSchema);

const jsonMetaSchema = z.union([
  typedJsonMetaSchema,
  refJsonMetaSchema,
  anyOfJsonMetaSchema,
  oneOfJsonMetaSchema,
  anyJsonMetaSchema,
]);

export type JsonSchema = z.infer<typeof jsonMetaSchema>;

const topLevelJsonMetaSchema = z.object({
  $defs: z.record(z.string(), jsonMetaSchema),
});

export type JsonSchemas = z.infer<typeof topLevelJsonMetaSchema>;

export async function getJsonSchemas(): Promise<JsonSchemas> {
  const response = await fetch(`${API_URL}/schemas`);
  const data = await response.json();
  const parsed = topLevelJsonMetaSchema.parse(data);
  return parsed;
}
