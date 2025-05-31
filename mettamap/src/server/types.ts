export type MapMetadata = {
  url: string;
};

export type MapData = {
  content: {
    frontmatter: string;
    data: string;
  };
}

export type MapIndex = Record<string, Record<string, string[]>>;
