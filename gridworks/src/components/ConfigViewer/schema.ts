import z from "zod/v4";

import mettaSchema from "../../lib/schemas.json" assert { type: "json" };

const commonJsonMetaSchema = {
  title: z.string().optional(),
  description: z.string().optional(),
};

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
      typeof anyOfJsonMetaSchema,
      typeof refJsonMetaSchema,
      typeof anyJsonMetaSchema,
      z.ZodLiteral<false>,
    ]
  > {
    return z.union([
      typedJsonMetaSchema,
      anyOfJsonMetaSchema,
      refJsonMetaSchema,
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
          typeof anyOfJsonMetaSchema,
          typeof refJsonMetaSchema,
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
        typeof anyOfJsonMetaSchema,
        typeof refJsonMetaSchema,
        typeof anyJsonMetaSchema,
      ]
    >
  > {
    return jsonMetaSchema.optional();
  },
});

type TypedJsonMetaSchema = z.ZodDiscriminatedUnion<
  [
    typeof stringJsonMetaSchema,
    typeof numberJsonMetaSchema,
    typeof integerJsonMetaSchema,
    typeof booleanJsonMetaSchema,
    typeof objectJsonMetaSchema,
    typeof arrayJsonMetaSchema,
  ]
>;

const typedJsonMetaSchema: TypedJsonMetaSchema = z.discriminatedUnion("type", [
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

const refJsonMetaSchema = z.object({
  $ref: z.string(),
});

const anyJsonMetaSchema = z.object(commonJsonMetaSchema);

const jsonMetaSchema = z.union([
  typedJsonMetaSchema,
  anyOfJsonMetaSchema,
  refJsonMetaSchema,
  anyJsonMetaSchema,
]);

export type JsonSchema = z.infer<typeof jsonMetaSchema>;

const topLevelJsonMetaSchema = z.object({
  $defs: z.record(z.string(), jsonMetaSchema),
});

const mettaSchemas = topLevelJsonMetaSchema.parse(mettaSchema);

export function getSchema(path: string, kind: string): JsonSchema | undefined {
  const defs = mettaSchemas.$defs;
  let currentType: JsonSchema = { $ref: "#/$defs/" + kind };
  const parts = path.split(".");
  for (const part of parts) {
    let nextType: JsonSchema | undefined;

    if ("$ref" in currentType) {
      const ref: string | undefined = currentType.$ref.split("/").pop()!;
      if (!ref) {
        console.warn("Unknown ref", currentType.$ref);
        return undefined;
      }
      if (!(ref in defs)) {
        console.warn("Unknown ref", currentType.$ref);
        return undefined;
      }
      currentType = defs[ref];
    }

    if (!("type" in currentType)) {
      return undefined;
    }
    if (currentType.type !== "object") {
      return undefined;
    }

    if (
      "properties" in currentType &&
      typeof currentType.properties === "object" &&
      currentType.properties !== null &&
      part in currentType.properties
    ) {
      // TypeScript bug: it doesn't infer `currentType.properties` correctly.
      // (I know it's because of TypeScript because I've seen this inference working sometimes on unrelated code changes.)
      nextType = (currentType.properties as Record<string, JsonSchema>)[part];
    }

    if (
      !nextType &&
      "additionalProperties" in currentType &&
      typeof currentType.additionalProperties === "object"
    ) {
      nextType = currentType.additionalProperties;
    }

    if (nextType) {
      currentType = nextType;
    } else {
      return undefined;
    }
  }
  return currentType;
}

export function getSchemaTypeStr(property: JsonSchema): string {
  if ("type" in property) {
    let typeStr: string = property.type;
    if (property.type === "array") {
      typeStr = `${typeStr}[${property.items ? getSchemaTypeStr(property.items) : "unknown"}]`;
    } else if (property.type === "object") {
      typeStr = `${typeStr}[${property.additionalProperties ? getSchemaTypeStr(property.additionalProperties) : "unknown"}]`;
    }
    return typeStr;
  } else if ("anyOf" in property) {
    return property.anyOf.map(getSchemaTypeStr).join(" | ");
  } else if ("$ref" in property) {
    return property.$ref.split("/").pop()!;
  } else {
    return "unknown";
  }
}
