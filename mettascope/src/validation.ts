import * as Common from './common.js'
import { Replay } from './replay.js'

const REQUIRED_KEYS = [
  'version',
  'numAgents',
  'maxSteps',
  'mapSize',
  'actionNames',
  'itemNames',
  'typeNames',
  'objects',
]

const OPTIONAL_KEYS = [
  'fileName',
  'groupNames',
  'rewardSharingMatrix',

  // frontend fields added by fixReplay() in replay.ts
  'actionImages',
  'agents',
  'objectImages',
  'resourceImages',
  'typeImages',
]

/** Interface for validation issues */
interface ValidationIssue {
  message: string
  field?: string
}

function validateType(value: any, expectedType: string, fieldName: string, issues: ValidationIssue[]): void {
  const actualType = Array.isArray(value) ? 'array' : typeof value

  if (expectedType === 'array' && !Array.isArray(value)) {
    issues.push({ message: `'${fieldName}' must be array, got ${actualType}`, field: fieldName })
  } else if (expectedType !== 'array' && typeof value !== expectedType) {
    issues.push({ message: `'${fieldName}' must be ${expectedType}, got ${actualType}`, field: fieldName })
  }
}

function validatePositiveInt(value: any, fieldName: string, issues: ValidationIssue[]): void {
  validateType(value, 'number', fieldName, issues)
  if (typeof value === 'number' && value <= 0) {
    issues.push({ message: `'${fieldName}' must be positive, got ${value}`, field: fieldName })
  }
}

function validateNonNegativeNumber(value: any, fieldName: string, issues: ValidationIssue[]): void {
  validateType(value, 'number', fieldName, issues)
  if (typeof value === 'number' && value < 0) {
    issues.push({ message: `'${fieldName}' must be non-negative, got ${value}`, field: fieldName })
  }
}

function validateStringList(lst: any, fieldName: string, issues: ValidationIssue[], allowEmpty: boolean = false): void {
  validateType(lst, 'array', fieldName, issues)

  if (Array.isArray(lst)) {
    if (!allowEmpty && lst.length === 0) {
      issues.push({ message: `'${fieldName}' must not be empty`, field: fieldName })
    }

    for (const item of lst) {
      if (typeof item !== 'string' || item === '') {
        issues.push({ message: `'${fieldName}' must contain non-empty strings`, field: fieldName })
        break
      }
    }
  }
}

function validateTimeSeries(data: any, fieldName: string, validator: (value: any) => boolean, issues: ValidationIssue[]): void {
  if (validator(data)) return

  if (!Array.isArray(data)) {
    issues.push({ message: `'${fieldName}' must be valid value or time series of [step, value] pairs`, field: fieldName })
    return
  }

  if (data.length === 0) return

  for (const item of data) {
    if (!Array.isArray(item) || item.length !== 2) {
      issues.push({ message: `'${fieldName}' time series items must be [step, value] pairs`, field: fieldName })
      return
    }

    if (typeof item[0] !== 'number' || item[0] < 0) {
      issues.push({ message: `'${fieldName}' time series step must be non-negative`, field: fieldName })
      return
    }

    if (!validator(item[1])) {
      issues.push({ message: `'${fieldName}' time series value is invalid`, field: fieldName })
      return
    }
  }

  if (data.length > 0 && data[0][0] !== 0) {
    issues.push({ message: `'${fieldName}' time series must start with step 0`, field: fieldName })
  }
}

function isNumber(value: any): boolean {
  return typeof value === 'number'
}

function isBoolean(value: any): boolean {
  return typeof value === 'boolean'
}

function isCoordinates(value: any): boolean {
  return Array.isArray(value) && value.length === 3 && value.every(coord => typeof coord === 'number')
}

function requireFields(obj: any, fields: string[], objName: string, issues: ValidationIssue[]): void {
  const missing = fields.filter(field => !(field in obj))
  if (missing.length > 0) {
    issues.push({ message: `${objName} missing required fields: ${missing.join(', ')}` })
  }
}

function validateLocation(location: any, objName: string, issues: ValidationIssue[]): void {
  validateTimeSeries(location, `${objName}.location`, isCoordinates, issues)
}

function validateInventoryList(inventoryList: any, fieldName: string, issues: ValidationIssue[]): void {
  validateType(inventoryList, 'array', fieldName, issues)

  if (Array.isArray(inventoryList)) {
    for (const pair of inventoryList) {
      if (!Array.isArray(pair) || pair.length !== 2) {
        issues.push({ message: `'${fieldName}' must contain [item_id, amount] pairs`, field: fieldName })
        return
      }

      if (typeof pair[0] !== 'number' || pair[0] < 0) {
        issues.push({ message: `'${fieldName}' item_id must be non-negative integer`, field: fieldName })
        return
      }

      if (typeof pair[1] !== 'number' || pair[1] < 0) {
        issues.push({ message: `'${fieldName}' amount must be non-negative number`, field: fieldName })
        return
      }
    }
  }
}

function validateInventoryFormat(inventory: any, fieldName: string, issues: ValidationIssue[]): void {
  if (!Array.isArray(inventory)) {
    validateType(inventory, 'array', fieldName, issues)
    return
  }

  if (inventory.length === 0) return

  // Single inventory list
  if (
    inventory.every(
      (item: any) =>
        Array.isArray(item) && item.length === 2 && typeof item[0] === 'number' && typeof item[1] === 'number'
    )
  ) {
    validateInventoryList(inventory, fieldName, issues)
    return
  }

  // Time series format
  for (const item of inventory) {
    if (!Array.isArray(item) || item.length !== 2) {
      issues.push({ message: `'${fieldName}' time series items must be [step, inventory_list] pairs`, field: fieldName })
      return
    }

    if (typeof item[0] !== 'number' || item[0] < 0) {
      issues.push({ message: `'${fieldName}' time series step must be non-negative`, field: fieldName })
      return
    }

    validateInventoryList(item[1], fieldName, issues)
  }
}

function validateObject(obj: any, objIndex: number, replayData: any, issues: ValidationIssue[]): void {
  const objName = `Object ${objIndex + 1}`

  const requiredFields = ['id', 'typeId', 'location', 'orientation', 'inventory', 'color']
  requireFields(obj, requiredFields, objName, issues)

  validateType(obj.id, 'number', `${objName}.id`, issues)
  validatePositiveInt(obj.id, `${objName}.id`, issues)

  validateType(obj.typeId, 'number', `${objName}.typeId`, issues)
  validateNonNegativeNumber(obj.typeId, `${objName}.typeId`, issues)
  if (obj.typeId >= replayData.typeNames.length) {
    issues.push({ message: `${objName}.typeId ${obj.typeId} out of range`, field: `${objName}.typeId` })
  }

  validateLocation(obj.location, objName, issues)
  validateTimeSeries(obj.orientation, `${objName}.orientation`, isNumber, issues)
  validateInventoryFormat(obj.inventory, `${objName}.inventory`, issues)
  validateTimeSeries(obj.color, `${objName}.color`, isNumber, issues)

  if (obj.isAgent || obj.agentId !== undefined) {
    validateAgentFields(obj, objName, replayData, issues)
  }
}

function validateAgentFields(obj: any, objName: string, replayData: any, issues: ValidationIssue[]): void {
  const agentFields = ['agentId', 'actionId', 'currentReward', 'totalReward', 'isFrozen']
  requireFields(obj, agentFields, objName, issues)

  validateType(obj.agentId, 'number', `${objName}.agentId`, issues)
  validateNonNegativeNumber(obj.agentId, `${objName}.agentId`, issues)
  if (obj.agentId >= replayData.numAgents) {
    issues.push({ message: `${objName}.agentId ${obj.agentId} out of range`, field: `${objName}.agentId` })
  }

  validateTimeSeries(obj.actionId, `${objName}.actionId`, isNumber, issues)
  validateTimeSeries(obj.currentReward, `${objName}.currentReward`, isNumber, issues)
  validateTimeSeries(obj.totalReward, `${objName}.totalReward`, isNumber, issues)
  validateTimeSeries(obj.isFrozen, `${objName}.isFrozen`, isBoolean, issues)
}

function validateReplaySchema(data: any, issues: ValidationIssue[]): void {
  const dataKeys = new Set(Object.keys(data))
  const missingKeys = REQUIRED_KEYS.filter((key) => !dataKeys.has(key))
  const allowedKeys = new Set([...REQUIRED_KEYS, ...OPTIONAL_KEYS])
  const unexpectedKeys = Array.from(dataKeys).filter((key) => !allowedKeys.has(key))

  if (missingKeys.length > 0) {
    issues.push({ message: `Missing required keys: ${missingKeys.sort().join(', ')}` })
  }

  if (unexpectedKeys.length > 0) {
    issues.push({ message: `Unexpected keys present: ${unexpectedKeys.sort().join(', ')}` })
  }

  if (data.version !== 2) {
    issues.push({ message: `'version' must equal 2, got ${data.version}`, field: 'version' })
  }

  validatePositiveInt(data.numAgents, 'numAgents', issues)
  validateNonNegativeNumber(data.maxSteps, 'maxSteps', issues)

  validateType(data.mapSize, 'array', 'mapSize', issues)
  if (data.mapSize && data.mapSize.length !== 2) {
    issues.push({ message: "'mapSize' must have exactly 2 dimensions", field: 'mapSize' })
  }
  if (data.mapSize && data.mapSize.length >= 1) {
    validatePositiveInt(data.mapSize[0], 'mapSize[0]', issues)
  }
  if (data.mapSize && data.mapSize.length >= 2) {
    validatePositiveInt(data.mapSize[1], 'mapSize[1]', issues)
  }

  validateStringList(data.actionNames, 'actionNames', issues)
  validateStringList(data.itemNames, 'itemNames', issues)
  validateStringList(data.typeNames, 'typeNames', issues)

  if (data.fileName !== undefined) {
    validateType(data.fileName, 'string', 'fileName', issues)
    if (data.fileName === '') {
      issues.push({ message: "'fileName' must be non-empty", field: 'fileName' })
    }
  }

  if (data.groupNames !== undefined) {
    validateStringList(data.groupNames, 'groupNames', issues, true)
  }

  if (data.rewardSharingMatrix !== undefined && data.rewardSharingMatrix.length > 0) {
    validateType(data.rewardSharingMatrix, 'array', 'rewardSharingMatrix', issues)
    const matrix = data.rewardSharingMatrix
    const numAgents = data.numAgents

    if (matrix && matrix.length !== numAgents) {
      issues.push({ message: `'rewardSharingMatrix' must have ${numAgents} rows`, field: 'rewardSharingMatrix' })
    }

    if (matrix) {
      for (let i = 0; i < matrix.length; i++) {
        validateType(matrix[i], 'array', `rewardSharingMatrix[${i}]`, issues)
        if (matrix[i] && matrix[i].length !== numAgents) {
          issues.push({ message: `'rewardSharingMatrix[${i}]' must have ${numAgents} columns`, field: `rewardSharingMatrix[${i}]` })
        }

        if (matrix[i]) {
          for (const value of matrix[i]) {
            if (typeof value !== 'number') {
              issues.push({ message: `'rewardSharingMatrix[${i}]' must contain numbers`, field: `rewardSharingMatrix[${i}]` })
              break
            }
          }
        }
      }
    }
  }

  validateType(data.objects, 'array', 'objects', issues)
  if (data.objects && data.objects.length === 0) {
    issues.push({ message: "'objects' must not be empty", field: 'objects' })
  }

  if (data.objects) {
    for (const obj of data.objects) {
      if (typeof obj !== 'object' || obj === null) {
        issues.push({ message: "'objects' must contain dictionaries", field: 'objects' })
        break
      }
    }

    let agentCount = 0
    for (let i = 0; i < data.objects.length; i++) {
      validateObject(data.objects[i], i, data, issues)
      if (data.objects[i].isAgent || data.objects[i].agentId !== undefined) {
        agentCount++
      }
    }

    if (agentCount !== data.numAgents) {
      issues.push({ message: `Expected ${data.numAgents} agents, found ${agentCount}`, field: 'objects' })
    }
  }
}

/** Displays validation issues in a modal */
function showValidationModal(issues: ValidationIssue[]): void {
  const title = `Validation Errors (${issues.length})`

  // Create a formatted message with all issues
  const formattedIssues = issues.map((issue, index) =>
    `${index + 1}. ${issue.message}`
  ).join('\n')

  const message = `Found ${issues.length} validation error(s):\n\n${formattedIssues}`

  Common.showModal('error', title, message)
}

export function validateReplayData(replay: Replay): void {
  // Skip validation if replay is not properly initialized yet
  if (!replay || !replay.objects || replay.objects.length === 0) {
    return
  }
  // skip validation for older format that we don't care about
  if (Common.state.replay.version < 2) {
    return
  }

  const issues: ValidationIssue[] = []
  validateReplaySchema(replay, issues)

  console.log("validation modal with", issues.length, "issues")
  if (issues.length > 0) {
    showValidationModal(issues)
  }
}

export function validateReplayStep(replayStep: any): void {
  if (Common.state.replay.version < 2) {
    return
  }

  const issues: ValidationIssue[] = []

  if (typeof replayStep.step !== 'number' || replayStep.step < 0) {
    issues.push({ message: `Invalid step number: ${replayStep.step}`, field: 'step' })
  }

  validateType(replayStep.objects, 'array', 'Replay step objects', issues)
  if (!Array.isArray(replayStep.objects)) {
    if (issues.length > 0) {
      showValidationModal(issues)
    }
    return
  }

      for (let i = 0; i < replayStep.objects.length; i++) {
      const obj = replayStep.objects[i]
      const prefix = `Step ${replayStep.step}, Object ${i + 1}:`

      if (typeof obj.id !== 'number' || obj.id < 1) {
        issues.push({ message: `${prefix} Invalid or missing ID`, field: `objects[${i}].id` })
      }

      if (typeof obj.type_id !== 'number' || obj.type_id < 0) {
        issues.push({ message: `${prefix} Invalid type ID`, field: `objects[${i}].type_id` })
      }

      if (obj.location !== undefined) {
        if (!Array.isArray(obj.location) || obj.location.length < 2) {
          issues.push({ message: `${prefix} Invalid location format`, field: `objects[${i}].location` })
        } else {
          const [x, y] = obj.location
          if (typeof x !== 'number' || typeof y !== 'number') {
            issues.push({ message: `${prefix} Location coordinates must be numbers`, field: `objects[${i}].location` })
          }
        }
      }

      if ('agent_id' in obj) {
        if (typeof obj.agent_id !== 'number' || obj.agent_id < 0) {
          issues.push({ message: `${prefix} Invalid agent ID`, field: `objects[${i}].agent_id` })
        }

        if (obj.action_id !== undefined && (typeof obj.action_id !== 'number' || obj.action_id < 0)) {
          issues.push({ message: `${prefix} Invalid action ID`, field: `objects[${i}].action_id` })
        }

        if (obj.current_reward !== undefined && typeof obj.current_reward !== 'number') {
          issues.push({ message: `${prefix} Current reward must be a number`, field: `objects[${i}].current_reward` })
        }

        if (obj.total_reward !== undefined && typeof obj.total_reward !== 'number') {
          issues.push({ message: `${prefix} Total reward must be a number`, field: `objects[${i}].total_reward` })
        }
      }

      if (obj.inventory !== undefined) {
        validateType(obj.inventory, 'array', `${prefix} Inventory`, issues)
        if (Array.isArray(obj.inventory)) {
          for (let j = 0; j < obj.inventory.length; j++) {
            const invItem = obj.inventory[j]
            if (
              !Array.isArray(invItem) ||
              invItem.length !== 2 ||
              typeof invItem[0] !== 'number' ||
              typeof invItem[1] !== 'number'
            ) {
              issues.push({ message: `${prefix} Invalid inventory item format at index ${j}`, field: `objects[${i}].inventory[${j}]` })
            }
          }
        }
      }
    }

    if (issues.length > 0) {
      showValidationModal(issues)
    }
}
