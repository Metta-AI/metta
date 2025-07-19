import {
  createLoader,
  createParser,
  parseAsArrayOf,
  parseAsInteger,
  parseAsString,
} from "nuqs/server";

export type FilterItem = {
  key: string;
  value: string;
};

const parseAsFilter = createParser<FilterItem>({
  parse(queryValue: string) {
    const [key, value] = queryValue.split("=");
    return { key, value };
  },
  serialize(value) {
    return `${value.key}=${value.value}`;
  },
});

export const parseFilterParam = parseAsArrayOf(parseAsFilter);

export const parseLimitParam = parseAsInteger.withDefault(10);

export const paramsLoader = createLoader({
  dir: parseAsString,
  filter: parseFilterParam,
  limit: parseLimitParam,
});
