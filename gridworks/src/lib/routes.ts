export function configsRoute() {
  return "/configs";
}

export function viewConfigRoute(name: string) {
  return `/configs/view/${name}`;
}

export function mapEditorRoute() {
  return "/map-editor";
}

export function cogamesRoute() {
  return "/cogames";
}

export function cogamesMissionsRoute() {
  return "/cogames/missions";
}

export function viewMissionRoute(name: string, variants: string[] = []) {
  return `/cogames/missions/view/${name}${variants.length > 0 ? `?variants=${variants.join(",")}` : ""}`;
}

export function viewMissionEnvRoute(name: string, variants: string[] = []) {
  return `${viewMissionRoute(name)}/env${variants.length > 0 ? `?variants=${variants.join(",")}` : ""}`;
}

export function viewMissionMapRoute(name: string, variants: string[] = []) {
  return `${viewMissionRoute(name)}/map${variants.length > 0 ? `?variants=${variants.join(",")}` : ""}`;
}
