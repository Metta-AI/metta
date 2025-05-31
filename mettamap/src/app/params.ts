import {
  createLoader,
  createParser,
  parseAsArrayOf,
  parseAsInteger,
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

export const parseLimitParam = parseAsInteger.withDefault(20);

export const paramsLoader = createLoader({
  filter: parseFilterParam,
  limit: parseLimitParam,
});
