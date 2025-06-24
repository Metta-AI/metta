/**
 * Demo mode is aimed to show cool thing about the replay without the any user interaction.
 * It could be used on a big TV to just play mettagrid in a loop.
 * The mode:
 * - Zooms in and out.
 * - Focuses agents when they are doing something interesting.
 * - Shows overall view.
 */

import { requestFrame } from './main.js'
import { focusFullMap } from './worldmap.js'
import { ui, state } from './common.js'

enum ShotType {
  FOCUS_AGENT = 'focus_agent',
  SHOW_OVERALL_VIEW = 'show_overall_view',
}

/** Unix epoch seconds sense 1970-01-01 00:00:00 UTC. */
function epochTime() {
  return Date.now() / 1000
}

class Shot {
  type: ShotType
  agentId: number
  duration: number
  startTime: number
  zooming: number

  constructor(type: ShotType, agentId: number, duration: number, zooming: number) {
    this.type = type
    this.agentId = agentId
    this.startTime = epochTime()
    this.duration = duration
    this.zooming = zooming
  }
}

var shot: Shot | null = null

export function initDemoMode() {
  // TODO: Implement demo mode.
}

export function startDemoMode() {
  state.demoMode = true
  requestFrame()
}

export function stopDemoMode() {
  state.demoMode = false
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
  if (!state.demoMode || state.replay == null) {
    return
  }
  state.isPlaying = true

  // Is the current shot over?
  if (shot == null || shot.startTime + shot.duration < epochTime()) {
    if (Math.random() < 0.1) {
      shot = new Shot(
        ShotType.SHOW_OVERALL_VIEW,
        0,
        3,
        0
      )
      state.selectedGridObject = null
      state.followSelection = false
      focusFullMap(ui.mapPanel)
    } else {
      shot = new Shot(
        ShotType.FOCUS_AGENT,
        Math.floor(Math.random() * state.replay.agents.length),
        3,
        choose([-0.1, 0.1])
      )
      state.selectedGridObject = state.replay.agents[shot.agentId]
      state.followSelection = true
    }
  }

  if (shot != null) {
    if (shot.type == ShotType.SHOW_OVERALL_VIEW) {
      // ?
    } else {
      let t = (epochTime() - shot.startTime) / shot.duration
      ui.mapPanel.zoomLevel = (0.3 - shot.zooming) + shot.zooming * easeInOut(t)
    }
  }

  requestFrame()
}
