import z from "zod/v4";

import jsonSchemas from "../../lib/schemas.json" assert { type: "json" };

const propertyCommonSchema = {
  title: z.string().optional(),
  description: z.string().optional(),
};

const stringPropertySchema = z.object({
  ...propertyCommonSchema,
  type: z.literal("string"),
});

const numberPropertySchema = z.object({
  ...propertyCommonSchema,
  type: z.literal("number"),
});

const integerPropertySchema = z.object({
  ...propertyCommonSchema,
  type: z.literal("integer"),
});

const booleanPropertySchema = z.object({
  ...propertyCommonSchema,
  type: z.literal("boolean"),
});

const objectPropertySchema = z.object({
  ...propertyCommonSchema,
  type: z.literal("object"),
  get additionalProperties(): z.ZodUnion<
    readonly [
      TypedPropertySchema,
      typeof anyOfPropertySchema,
      typeof refPropertySchema,
    ]
  > {
    return propertySchema;
  },
});

const arrayPropertySchema = z.object({
  ...propertyCommonSchema,
  type: z.literal("array"),
  get items(): z.ZodOptional<
    z.ZodUnion<
      readonly [
        TypedPropertySchema,
        typeof anyOfPropertySchema,
        typeof refPropertySchema,
      ]
    >
  > {
    return propertySchema.optional();
  },
});

type TypedPropertySchema = z.ZodDiscriminatedUnion<
  [
    typeof stringPropertySchema,
    typeof numberPropertySchema,
    typeof integerPropertySchema,
    typeof booleanPropertySchema,
    typeof objectPropertySchema,
    typeof arrayPropertySchema,
  ]
>;

const typedPropertySchema: TypedPropertySchema = z.discriminatedUnion("type", [
  stringPropertySchema,
  numberPropertySchema,
  integerPropertySchema,
  booleanPropertySchema,
  objectPropertySchema,
  arrayPropertySchema,
]);

const anyOfPropertySchema = z.object({
  ...propertyCommonSchema,
  get anyOf(): z.ZodArray<
    z.ZodUnion<
      [TypedPropertySchema, typeof refPropertySchema, typeof anyPropertySchema]
    >
  > {
    return z.array(
      z.union([typedPropertySchema, refPropertySchema, anyPropertySchema])
    );
  },
});

const refPropertySchema = z.object({
  $ref: z.string(),
});

const anyPropertySchema = z.object({});

const propertySchema = z.union([
  typedPropertySchema,
  anyOfPropertySchema,
  refPropertySchema,
  anyPropertySchema,
]);

export type SchemaProperty = z.infer<typeof propertySchema>;

const jsonSchemasMetaSchema = z.object({
  $defs: z.record(
    z.string(),
    z.object({
      properties: z.record(z.string(), propertySchema),
    })
  ),
});

const typedSchemas = jsonSchemasMetaSchema.parse(jsonSchemas);

export function getSchemaProperty(
  path: string,
  kind: string
): SchemaProperty | undefined {
  const defs = typedSchemas.$defs;
  let currentKind = kind;
  const parts = path.split(".");
  let property: SchemaProperty | undefined = undefined;
  for (const part of parts) {
    const def = defs[currentKind];
    if (!def) {
      console.warn("Unknown kind", kind);
      return undefined;
    }
    property = def.properties[part];
    if (!property) {
      return undefined;
    }
    if ("$ref" in property) {
      currentKind = property.$ref.split("/").pop()!;
    } else {
      currentKind = "UNKNOWN";
    }
  }
  const parsed = propertySchema.safeParse(property);
  if (!parsed.success) {
    console.error(kind, path, parsed.error);
  }
  return parsed.success ? parsed.data : undefined;
}

export function getPropertyTypeStr(property: SchemaProperty): string {
  if ("type" in property) {
    let typeStr: string = property.type;
    if (property.type === "array") {
      typeStr = `${typeStr}[${property.items ? getPropertyTypeStr(property.items) : "unknown"}]`;
    } else if (property.type === "object") {
      typeStr = `${typeStr}[${getPropertyTypeStr(property.additionalProperties)}]`;
    }
    return typeStr;
  } else if ("anyOf" in property) {
    return property.anyOf.map(getPropertyTypeStr).join(" | ");
  } else if ("$ref" in property) {
    return property.$ref.split("/").pop()!;
  } else {
    return "unknown";
  }
}
