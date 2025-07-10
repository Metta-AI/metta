/**
 * Demo mode is aimed to show cool things about the replay without any user interaction.
 * It could be used on a big TV to just play mettagrid in a loop.
 * The mode:
 * - Zooms in and out.
 * - Focuses agents when they are doing something interesting.
 * - Shows overall view.
 */

import { requestFrame, setIsPlaying } from './main.js'
import { focusFullMap, focusMap } from './worldmap.js'
import { ui, state } from './common.js'
import * as Common from './common.js'

enum ShotType {
  FOCUS_AGENT = 'focus_agent',
  SHOW_OVERALL_VIEW = 'show_overall_view',
}

/** Unix epoch seconds since 1970-01-01 00:00:00 UTC. */
function epochTime() {
  return Date.now() / 1000
}

/** Series of viewing parameters, like duration and what to focus on. */
class Shot {
  type: ShotType
  agentId: number
  duration: number
  startTime: number
  zooming: number
  zoomLevel: number

  constructor(type: ShotType, agentId: number, duration: number, zooming: number) {
    this.type = type
    this.agentId = agentId
    this.startTime = epochTime()
    this.duration = duration
    this.zooming = zooming
    this.zoomLevel = 0.3
  }
}

/** Current active shot. */
var shot: Shot | null = null

export function initDemoMode() {
}

export function startDemoMode() {
  state.demoMode = true
  requestFrame()
}

export function stopDemoMode() {
  state.demoMode = false
  setIsPlaying(false)
  requestFrame()
}

/** Easing function that eases in and out. */
function easeInOut(time: number) {
  return time < 0.5 ? 2 * time * time : -1 + (4 - 2 * time) * time
}

/** Chooses a random value from an array. */
function choose<T>(values: T[]): T {
  return values[Math.floor(Math.random() * values.length)]
}

export function doDemoMode() {
  if (!state.demoMode || state.replay == null || !state.replayHelper) {
    return
  }
  state.isPlaying = true

  // Is the current shot over?
  if (shot == null || shot.startTime + shot.duration < epochTime()) {
    if (Math.random() < 0.1) {
      // Create a shot that just shows the overall view.
      shot = new Shot(
        ShotType.SHOW_OVERALL_VIEW,
        0,
        3,
        0
      )
      state.selectedGridObject = null
      state.followSelection = false
      focusFullMap(ui.mapPanel)
      shot.zooming = choose([-0.2, 0.2]) * ui.mapPanel.zoomLevel
      shot.zoomLevel = ui.mapPanel.zoomLevel * 1.2
    } else {
      // Find an agent that will do some thing interesting soon.
      var agentId = Math.floor(Math.random() * state.replayHelper.agents.length)
      for (let i = 0; i < state.replayHelper.agents.length; i++) {
        let agent = state.replayHelper.agents[i]
        if (!agent) continue
        let actionFound = false
        for (let j = 0; j < 10; j++) {
          let action = agent.actionId.get(state.step + j)
          if (action == null) {
            continue
          }
          const actionName = state.replay.actionNames[action as number]
          let actionSuccess = agent.actionSuccess.get(state.step + j)
          if (
            actionName != 'noop' &&
            actionName != 'rotate' &&
            actionName != 'move' &&
            actionName != 'change_color' &&
            actionName != 'change_shape' &&
            actionSuccess
          ) {
            agentId = i
            actionFound = true
            break
          }
        }
        if (actionFound) {
          break
        }
      }

      // Create a new shot that focuses on the agent.
      shot = new Shot(
        ShotType.FOCUS_AGENT,
        agentId,
        3,
        choose([-0.1, 0.1])
      )
      state.selectedGridObject = state.replayHelper.agents[shot.agentId]
      state.followSelection = true
      focusMap(0, 0, 11 * Common.TILE_SIZE, 11 * Common.TILE_SIZE)
      shot.zoomLevel = ui.mapPanel.zoomLevel * 1.5
    }
  }

  let t = (epochTime() - shot.startTime) / shot.duration
  if (shot != null) {
    ui.mapPanel.zoomLevel = (shot.zoomLevel - shot.zooming) + shot.zooming * easeInOut(t)
  }

  requestFrame()
}
