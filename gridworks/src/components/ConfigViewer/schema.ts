import { use } from "react";

import { JsonSchema, OneOfJsonMetaSchema } from "@/lib/api/schemas";

import { JsonSchemasContext } from "../global-contexts/JsonSchemasContext";
import { ConfigNode } from "./utils";
import { YamlContext } from "./YamlContext";

// Resolve { $ref: "#/$defs/..." } to the actual schema.
function useResolveType(): (type: JsonSchema) => JsonSchema | undefined {
  const { schemas } = use(JsonSchemasContext);

  const resolveType = (type: JsonSchema): JsonSchema | undefined => {
    if ("$ref" in type) {
      const ref: string | undefined = type.$ref.split("/").pop()!;
      if (!ref) {
        console.warn("Unknown ref", type.$ref);
        return undefined;
      }
      if (!(ref in schemas.$defs)) {
        console.warn("Unknown ref", type.$ref);
        return undefined;
      }
      return { ...schemas.$defs[ref], isTopLevelDef: true };
    }
    return type;
  };
  return resolveType;
}

// Convert a top-level config kind string like "List[SimulationConfig]" to a JSON schema.
function parseKind(kind: string): JsonSchema {
  const listMatch = kind.match(/^List\[(\w+)\]$/);
  if (listMatch) {
    return { type: "array", items: parseKind(listMatch[1]) };
  }
  return { $ref: "#/$defs/" + kind };
}

type NodeSchemaLookupDebugInfo = Record<string, never>;

type NodeSchemaResult = {
  schema: JsonSchema | undefined;
  debugInfo: NodeSchemaLookupDebugInfo; // could be used for debugging if needed, for example storing the full trace the useNodeSchema steps
};

function isPolymorphicType(currentType: JsonSchema): boolean {
  // Heuristics when we know that Pydantic does custom `@field_validator` with polymorphic dispatch based on `type: module.path.ClassName`
  if ("title" in currentType && currentType.title === "MapBuilderConfig[Any]") {
    return true;
  }

  if (
    "anyOf" in currentType &&
    currentType.anyOf.some(
      (c) => "$ref" in c && c["$ref"] === "#/$defs/MapBuilderConfig_Any_"
    )
  ) {
    return true;
  }
  return false;
}

function useGetValueByPath(): (path: string[]) => unknown {
  const { topValue } = use(YamlContext);
  const getValueByPath = (path: string[]): unknown => {
    let currentValue: unknown = topValue;
    for (const part of path) {
      if (typeof currentValue !== "object" || currentValue === null) {
        return null;
      }
      if (Array.isArray(currentValue) && part.match(/^\d+$/)) {
        currentValue = currentValue[Number(part)];
      } else {
        currentValue = (currentValue as Record<string, unknown>)[part];
      }
    }
    return currentValue ?? null;
  };
  return getValueByPath;
}

// Get the JSON schema for a given config value node.
export function useNodeSchema(node: ConfigNode): NodeSchemaResult {
  const debugInfo: NodeSchemaLookupDebugInfo = {};

  const { kind } = use(YamlContext);
  const { schemas } = use(JsonSchemasContext);
  const getValueByPath = useGetValueByPath();
  const resolveType = useResolveType();

  if (!kind) {
    return { schema: undefined, debugInfo };
  }

  const initialType = resolveType(parseKind(kind));
  if (!initialType) {
    return { schema: undefined, debugInfo };
  }
  let currentType: JsonSchema = initialType;

  for (let i = 0; i < node.path.length; i++) {
    const part = node.path[i];
    let nextType: JsonSchema | undefined;

    if (!("type" in currentType)) {
      return { schema: undefined, debugInfo };
    }

    if (part.match(/^\d+$/)) {
      if (currentType.type !== "array" || !currentType.items) {
        return { schema: undefined, debugInfo };
      }
      nextType = currentType.items;
      if (!nextType) {
        return { schema: undefined, debugInfo };
      }
      currentType = resolveType(nextType)!;
      continue;
    }

    if (currentType.type !== "object") {
      return { schema: undefined, debugInfo };
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
      currentType = resolveType(nextType)!;
    } else {
      return { schema: undefined, debugInfo };
    }

    // Resolve polymorphic types
    if ("oneOf" in currentType) {
      const oneOfType = currentType as OneOfJsonMetaSchema;
      const discriminator = oneOfType.discriminator;
      if (!discriminator) {
        // can't interpret non-discriminated oneOf
        return { schema: undefined, debugInfo };
      }
      const currentValue = getValueByPath(node.path.slice(0, i + 1));
      if (
        typeof currentValue === "object" &&
        currentValue !== null &&
        "type" in currentValue
      ) {
        const discriminatorValueStr = String(currentValue["type"]);
        const mappedType = discriminator.mapping[discriminatorValueStr];
        if (!mappedType) {
          return { schema: undefined, debugInfo };
        }
        const discriminatedType = resolveType({ $ref: mappedType });
        if (discriminatedType) {
          currentType = discriminatedType;
        } else {
          console.warn(
            "Unknown discriminated type, keeping original type",
            discriminatorValueStr
          );
        }
      }
    } else if (isPolymorphicType(currentType)) {
      const currentValue = getValueByPath(node.path.slice(0, i + 1));
      if (
        currentValue &&
        typeof currentValue === "object" &&
        "type" in currentValue
      ) {
        const typeModule = String(currentValue["type"]);
        const typesToTry = [
          // FQDN
          typeModule.split(".").join("__") + "__Config",
          // Short name - used by scenes
          typeModule.split(".").pop()! + "Config",
        ];
        let found = false;
        for (const type of typesToTry) {
          if (type in schemas.$defs) {
            currentType = resolveType({ $ref: "#/$defs/" + type })!;
            found = true;
            break;
          }
        }
        if (!found) {
          console.warn("Unknown type - can't find definition", typeModule);
        }
      }
    }
  }
  return { schema: currentType, debugInfo };
}

// Pretty python-like type string for a JSON schema.
export function getSchemaTypeStr(property: JsonSchema): string {
  if (
    "isTopLevelDef" in property &&
    "title" in property &&
    property.isTopLevelDef &&
    property.title
  ) {
    return property.title;
  }
  if ("type" in property) {
    let typeStr: string = property.type;
    if (property.type === "array") {
      typeStr = `list[${property.items ? getSchemaTypeStr(property.items) : "unknown"}]`;
    } else if (property.type === "object") {
      typeStr = `dict[str, ${property.additionalProperties ? getSchemaTypeStr(property.additionalProperties) : "unknown"}]`;
      // prefer python-like names
    } else if (property.type === "integer") {
      typeStr = "int";
    } else if (property.type === "number") {
      typeStr = "float";
    } else if (property.type === "boolean") {
      typeStr = "bool";
    }
    return typeStr;
  } else if ("anyOf" in property) {
    return property.anyOf.map(getSchemaTypeStr).join(" | ");
  } else if ("oneOf" in property) {
    return property.oneOf.map(getSchemaTypeStr).join(" | ");
  } else if ("$ref" in property) {
    return property.$ref.split("/").pop()!;
  } else {
    return "unknown";
  }
}
