// This file defines object right or control click menu.

import { find, findIn, onEvent, showMenu, findAttr } from "./htmlutils.js";
import { state, ui } from "./common.js";

var objectMenu = find("#object-menu");

export function initObjectMenu() {
  console.log("Object menu", objectMenu);
  objectMenu.classList.add("hidden");
}

onEvent("click", ".infopanel .memory", (target: HTMLElement, e: Event) => {
  console.log("Memory clicked");
  showMenu(target, objectMenu);
});

onEvent("click", ".infopanel .set-memory-to-0", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Clearing memory to 0");
  state.ws.send(JSON.stringify({
    type: "clear_memory",
    what: "0",
    agent_id: agentId
  }));
})

onEvent("click", ".infopanel .set-memory-to-1", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Clearing memory to 1");
  state.ws.send(JSON.stringify({
    type: "clear_memory",
    what: "1",
    agent_id: agentId
  }));
})

onEvent("click", ".infopanel .set-memory-to-random", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Clearing memory to random");
  state.ws.send(JSON.stringify({
    type: "clear_memory",
    what: "random",
    agent_id: agentId
  }));
})
