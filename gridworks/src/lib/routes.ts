export function configsRoute() {
  return "/configs";
}

export function viewConfigRoute(name: string) {
  return `/configs/view/${name}`;
}

export function mapEditorRoute() {
  return "/map-editor";
}

export function storedMapsRoute() {
  return "/stored-maps";
}

export function viewStoredMapsDirRoute(url: string) {
  return `/stored-maps/dir?dir=${url}`;
}

export function viewStoredMapRoute(url: string) {
  return `/stored-maps/view?map=${url}`;
}
