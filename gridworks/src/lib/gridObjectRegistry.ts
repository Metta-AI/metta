// based on mettascope's colorFromId
function colorFromId(agentId: number) {
  const n = agentId + Math.PI + Math.E + Math.SQRT2;
  return {
    r: (n * Math.PI) % 1.0,
    g: (n * Math.E) % 1.0,
    b: (n * Math.SQRT2) % 1.0,
  };
}

type Modulate = { r: number; g: number; b: number };

const red: Modulate = { r: 1, g: 0, b: 0 };
const blue: Modulate = { r: 0, g: 0, b: 1 };
const green: Modulate = { r: 0, g: 1, b: 0 };

type ObjectLayer = {
  tile: string;
  modulate?: Modulate;
};

type GridObjectType = {
  name: string;
  hotkey?: string;
  layers: ObjectLayer[];
};

function basicObject(name: string, hotkey: string): GridObjectType {
  if (hotkey.length !== 1) {
    throw new Error(`Invalid hotkey: ${hotkey}`);
  }
  return { name, layers: [{ tile: name }], hotkey };
}

function coloredObjects(name: string, hotkey: string): GridObjectType[] {
  return [
    {
      name: `${name}_red`,
      layers: [{ tile: name }, { tile: `${name}.color`, modulate: red }],
      hotkey,
    },
    {
      name: `${name}_blue`,
      layers: [{ tile: name }, { tile: `${name}.color`, modulate: blue }],
      hotkey,
    },
    {
      name: `${name}_green`,
      layers: [{ tile: name }, { tile: `${name}.color`, modulate: green }],
      hotkey,
    },
  ];
}

const gridObjectTypes: GridObjectType[] = [
  { name: "empty", layers: [], hotkey: "." },
  basicObject("wall", "w"),
  basicObject("block", "b"),
  basicObject("altar", "a"),
  basicObject("armory", "o"),
  basicObject("factory", "f"),
  basicObject("lab", "l"),
  basicObject("lasery", "s"),
  basicObject("temple", "t"),
  ...coloredObjects("mine", "m"),
  ...coloredObjects("generator", "n"),
  { name: "agent.agent", layers: [{ tile: "agent" }], hotkey: "g" },
  {
    name: "agent.team_1",
    layers: [{ tile: "agent", modulate: colorFromId(0) }],
    hotkey: "g",
  },
  {
    name: "agent.team_2",
    layers: [{ tile: "agent", modulate: colorFromId(1) }],
    hotkey: "g",
  },
  {
    name: "agent.team_3",
    layers: [{ tile: "agent", modulate: colorFromId(2) }],
    hotkey: "g",
  },
  {
    name: "agent.team_4",
    layers: [{ tile: "agent", modulate: colorFromId(3) }],
    hotkey: "g",
  },
  {
    name: "agent.prey",
    layers: [{ tile: "agent", modulate: green }],
    hotkey: "g",
  },
  {
    name: "agent.predator",
    layers: [{ tile: "agent", modulate: red }],
    hotkey: "g",
  },
];

const objects: Record<string, GridObjectType> = {};
for (const object of gridObjectTypes) {
  objects[object.name] = object;
}

const hotkeyToObjects: Record<string, GridObjectType[]> = {};
for (const object of gridObjectTypes) {
  if (object.hotkey) {
    hotkeyToObjects[object.hotkey] = [
      ...(hotkeyToObjects[object.hotkey] || []),
      object,
    ];
  }
}

export const gridObjectRegistry = {
  objectNames: gridObjectTypes.map((type) => type.name),
  getLayers(name: string) {
    const object = objects[name];
    if (!object) {
      throw new Error(`Unknown object name: ${name}`);
    }
    return object.layers;
  },
  objectByName(name: string): GridObjectType | undefined {
    return objects[name];
  },
  objectByHotkey(
    hotkey: string,
    currentObject?: string
  ): GridObjectType | undefined {
    const objects = hotkeyToObjects[hotkey];
    if (!objects) {
      return undefined;
    }
    const sameGroup = !!objects.filter(
      (object) => object.name === currentObject
    );
    if (sameGroup) {
      // jump to next object in group
      const index = objects.findIndex(
        (object) => object.name === currentObject
      );
      return objects[(index + 1) % objects.length];
    } else {
      return objects[0];
    }
  },
  allHotkeys() {
    return Object.keys(hotkeyToObjects);
  },
};
