import { find, finds, removeChildren, walkUpAttribute } from "./htmlutils.js";
import { state, setFollowSelection } from "./common.js";
import { getAttr } from "./replay.js";

const agentTable = find("#agent-table");
const header = find("#agent-table .header");
const fieldHeaderTemplate = finds("#agent-table .field-header")[0];
const rowTemplate = finds("#agent-table .row")[0];
const cellTemplate = finds("#agent-table .cell")[0];

export function initAgentTable() {
  removeChildren(rowTemplate);

  agentTable.addEventListener("click", (event) => {
    // Get the element that was actually clicked
    let target = event.target as HTMLElement;
    let agentId = walkUpAttribute(target, "data-agent-id");
    if (agentId != null) {
      for (let i = 0; i < state.replay.grid_objects.length; i++) {
        let gridObject = state.replay.grid_objects[i];
        if (getAttr(gridObject, "agent_id") == agentId) {
          state.selectedGridObject = gridObject;
          setFollowSelection(true);
          break;
        }
      }
    }
  });
}

export function updateAgentTable() {
  removeChildren(agentTable);
  agentTable.appendChild(header);

  let agents = state.replay.agents.sort((a: any, b: any) => {
    return getAttr(b, "total_reward") - getAttr(a, "total_reward");
  });

  for (let i = 0; i < agents.length; i++) {
    let agent = agents[i];
    let row = rowTemplate.cloneNode(true) as HTMLElement;
    row.setAttribute("data-agent-id", getAttr(agent, "agent_id").toString());
    header.appendChild(row);

    let cell = cellTemplate.cloneNode(true) as HTMLElement;
    cell.children[0].textContent = getAttr(agent, "agent_id").toString();
    row.appendChild(cell);

    cell = cellTemplate.cloneNode(true) as HTMLElement;
    cell.children[0].textContent = getAttr(agent, "total_reward").toFixed(3);
    row.appendChild(cell);

    cell = cellTemplate.cloneNode(true) as HTMLElement;
    cell.children[0].textContent = getAttr(agent, "agent:inv:heart").toString();
    row.appendChild(cell);

    cell = cellTemplate.cloneNode(true) as HTMLElement;
    cell.children[0].textContent = getAttr(agent, "agent:inv:ore.red").toString();
    row.appendChild(cell);

    agentTable.appendChild(row);
  }
}
