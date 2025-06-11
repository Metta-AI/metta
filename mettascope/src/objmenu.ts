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
  let agentId = findAttr(target, "data-agent-id");
  objectMenu.setAttribute("data-agent-id", agentId);
  showMenu(target, objectMenu);
});

onEvent("click", "#object-menu .set-memory-to-0", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Clearing memory to 0");
  state.ws.send(JSON.stringify({
    type: "clear_memory",
    what: "0",
    agent_id: agentId
  }));
})

onEvent("click", "#object-menu .set-memory-to-1", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Clearing memory to 1");
  state.ws.send(JSON.stringify({
    type: "clear_memory",
    what: "1",
    agent_id: agentId
  }));
})

onEvent("click", "#object-menu .set-memory-to-random", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Clearing memory to random");
  state.ws.send(JSON.stringify({
    type: "clear_memory",
    what: "random",
    agent_id: agentId
  }));
})

onEvent("click", "#object-menu .copy-memory", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Copying memory");
  state.ws.send(JSON.stringify({
    type: "copy_memory",
    agent_id: agentId
  }));
})

onEvent("click", "#object-menu .paste-memory", (target: HTMLElement, e: Event) => {
  if (state.ws == null) return;
  let agentId = parseInt(findAttr(target, "data-agent-id"));
  console.log("Pasting memory");
  state.ws.send(JSON.stringify({
    type: "paste_memory",
    agent_id: agentId,
    memory: JSON.parse(localStorage.getItem("memory") || "[[], []]")
  }));
})
