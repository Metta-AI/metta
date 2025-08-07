import { showWarningToast } from './warning_toast.js'
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

function validateType(value: any, expectedType: string, fieldName: string): void {
  const actualType = Array.isArray(value) ? 'array' : typeof value

  if (expectedType === 'array' && !Array.isArray(value)) {
    throw new Error(`'${fieldName}' must be array, got ${actualType}`)
  } else if (expectedType !== 'array' && typeof value !== expectedType) {
    throw new Error(`'${fieldName}' must be ${expectedType}, got ${actualType}`)
  }
}

function validatePositiveInt(value: any, fieldName: string): void {
  validateType(value, 'number', fieldName)
  if (value <= 0) {
    throw new Error(`'${fieldName}' must be positive, got ${value}`)
  }
}

function validateNonNegativeNumber(value: any, fieldName: string): void {
  validateType(value, 'number', fieldName)
  if (value < 0) {
    throw new Error(`'${fieldName}' must be non-negative, got ${value}`)
  }
}

function validateStringList(lst: any, fieldName: string, allowEmpty: boolean = false): void {
  validateType(lst, 'array', fieldName)

  if (!allowEmpty && lst.length === 0) {
    throw new Error(`'${fieldName}' must not be empty`)
  }

  for (const item of lst) {
    if (typeof item !== 'string' || item === '') {
      throw new Error(`'${fieldName}' must contain non-empty strings`)
    }
  }
}

function validateTimeSeries(data: any, fieldName: string, validator: (value: any) => boolean): void {
  if (validator(data)) return

  if (!Array.isArray(data)) {
    throw new Error(`'${fieldName}' must be valid value or time series of [step, value] pairs`)
  }

  if (data.length === 0) return

  for (const item of data) {
    if (!Array.isArray(item) || item.length !== 2) {
      throw new Error(`'${fieldName}' time series items must be [step, value] pairs`)
    }

    if (typeof item[0] !== 'number' || item[0] < 0) {
      throw new Error(`'${fieldName}' time series step must be non-negative`)
    }

    if (!validator(item[1])) {
      throw new Error(`'${fieldName}' time series value is invalid`)
    }
  }

  if (data.length > 0 && data[0][0] !== 0) {
    throw new Error(`'${fieldName}' time series must start with step 0`)
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

function requireFields(obj: any, fields: string[], objName: string): void {
  const missing = fields.filter(field => !(field in obj))
  if (missing.length > 0) {
    throw new Error(`${objName} missing required fields: ${missing.join(', ')}`)
  }
}

function validateLocation(location: any, objName: string): void {
  validateTimeSeries(location, `${objName}.location`, isCoordinates)
}

function validateInventoryList(inventoryList: any, fieldName: string): void {
  validateType(inventoryList, 'array', fieldName)

  for (const pair of inventoryList) {
    if (!Array.isArray(pair) || pair.length !== 2) {
      throw new Error(`'${fieldName}' must contain [item_id, amount] pairs`)
    }

    if (typeof pair[0] !== 'number' || pair[0] < 0) {
      throw new Error(`'${fieldName}' item_id must be non-negative integer`)
    }

    if (typeof pair[1] !== 'number' || pair[1] < 0) {
      throw new Error(`'${fieldName}' amount must be non-negative number`)
    }
  }
}

function validateInventoryFormat(inventory: any, fieldName: string): void {
  if (!Array.isArray(inventory)) {
    validateType(inventory, 'array', fieldName)
  }

  if (inventory.length === 0) return

  // Single inventory list
  if (
    inventory.every(
      (item: any) =>
        Array.isArray(item) && item.length === 2 && typeof item[0] === 'number' && typeof item[1] === 'number'
    )
  ) {
    validateInventoryList(inventory, fieldName)
    return
  }

  // Time series format
  for (const item of inventory) {
    if (!Array.isArray(item) || item.length !== 2) {
      throw new Error(`'${fieldName}' time series items must be [step, inventory_list] pairs`)
    }

    if (typeof item[0] !== 'number' || item[0] < 0) {
      throw new Error(`'${fieldName}' time series step must be non-negative`)
    }

    validateInventoryList(item[1], fieldName)
  }
}

function validateObject(obj: any, objIndex: number, replayData: any): void {
  const objName = `Object ${objIndex + 1}`

  const requiredFields = ['id', 'typeId', 'location', 'orientation', 'inventory', 'color']
  requireFields(obj, requiredFields, objName)

  validateType(obj.id, 'number', `${objName}.id`)
  validatePositiveInt(obj.id, `${objName}.id`)

  validateType(obj.typeId, 'number', `${objName}.typeId`)
  validateNonNegativeNumber(obj.typeId, `${objName}.typeId`)
  if (obj.typeId >= replayData.typeNames.length) {
    throw new Error(`${objName}.typeId ${obj.typeId} out of range`)
  }

  validateLocation(obj.location, objName)
  validateTimeSeries(obj.orientation, `${objName}.orientation`, isNumber)
  validateInventoryFormat(obj.inventory, `${objName}.inventory`)
  validateTimeSeries(obj.color, `${objName}.color`, isNumber)

  if (obj.isAgent || obj.agentId !== undefined) {
    validateAgentFields(obj, objName, replayData)
  }
}

function validateAgentFields(obj: any, objName: string, replayData: any): void {
  const agentFields = ['agentId', 'actionId', 'currentReward', 'totalReward', 'isFrozen']
  requireFields(obj, agentFields, objName)

  validateType(obj.agentId, 'number', `${objName}.agentId`)
  validateNonNegativeNumber(obj.agentId, `${objName}.agentId`)
  if (obj.agentId >= replayData.numAgents) {
    throw new Error(`${objName}.agentId ${obj.agentId} out of range`)
  }

  validateTimeSeries(obj.actionId, `${objName}.actionId`, isNumber)
  validateTimeSeries(obj.currentReward, `${objName}.currentReward`, isNumber)
  validateTimeSeries(obj.totalReward, `${objName}.totalReward`, isNumber)
  validateTimeSeries(obj.isFrozen, `${objName}.isFrozen`, isBoolean)
}

function validateReplaySchema(data: any): void {
  const dataKeys = new Set(Object.keys(data))
  const missingKeys = REQUIRED_KEYS.filter((key) => !dataKeys.has(key))
  const allowedKeys = new Set([...REQUIRED_KEYS, ...OPTIONAL_KEYS])
  const unexpectedKeys = Array.from(dataKeys).filter((key) => !allowedKeys.has(key))

  if (missingKeys.length > 0) {
    throw new Error(`Missing required keys: ${missingKeys.sort().join(', ')}`)
  }

  if (unexpectedKeys.length > 0) {
    throw new Error(`Unexpected keys present: ${unexpectedKeys.sort().join(', ')}`)
  }

  if (data.version !== 2) {
    throw new Error(`'version' must equal 2, got ${data.version}`)
  }

  validatePositiveInt(data.numAgents, 'numAgents')
  validateNonNegativeNumber(data.maxSteps, 'maxSteps')

  validateType(data.mapSize, 'array', 'mapSize')
  if (data.mapSize.length !== 2) {
    throw new Error("'mapSize' must have exactly 2 dimensions")
  }
  validatePositiveInt(data.mapSize[0], 'mapSize[0]')
  validatePositiveInt(data.mapSize[1], 'mapSize[1]')

  validateStringList(data.actionNames, 'actionNames')
  validateStringList(data.itemNames, 'itemNames')
  validateStringList(data.typeNames, 'typeNames')

  if (data.fileName !== undefined) {
    validateType(data.fileName, 'string', 'fileName')
    if (data.fileName === '') {
      throw new Error("'fileName' must be non-empty")
    }
  }

  if (data.groupNames !== undefined) {
    validateStringList(data.groupNames, 'groupNames', true)
  }

  if (data.rewardSharingMatrix !== undefined && data.rewardSharingMatrix.length > 0) {
    validateType(data.rewardSharingMatrix, 'array', 'rewardSharingMatrix')
    const matrix = data.rewardSharingMatrix
    const numAgents = data.numAgents

    if (matrix.length !== numAgents) {
      throw new Error(`'rewardSharingMatrix' must have ${numAgents} rows`)
    }

    for (let i = 0; i < matrix.length; i++) {
      validateType(matrix[i], 'array', `rewardSharingMatrix[${i}]`)
      if (matrix[i].length !== numAgents) {
        throw new Error(`'rewardSharingMatrix[${i}]' must have ${numAgents} columns`)
      }

      for (const value of matrix[i]) {
        if (typeof value !== 'number') {
          throw new Error(`'rewardSharingMatrix[${i}]' must contain numbers`)
        }
      }
    }
  }

  validateType(data.objects, 'array', 'objects')
  if (data.objects.length === 0) {
    throw new Error("'objects' must not be empty")
  }

  for (const obj of data.objects) {
    if (typeof obj !== 'object' || obj === null) {
      throw new Error("'objects' must contain dictionaries")
    }
  }

  let agentCount = 0
  for (let i = 0; i < data.objects.length; i++) {
    validateObject(data.objects[i], i, data)
    if (data.objects[i].isAgent || data.objects[i].agentId !== undefined) {
      agentCount++
    }
  }

  if (agentCount !== data.numAgents) {
    throw new Error(`Expected ${data.numAgents} agents, found ${agentCount}`)
  }
}

export function validateReplayData(replay: Replay): void {
  try {
    // Skip validation if replay is not properly initialized yet
    if (!replay || !replay.objects || replay.objects.length === 0) {
      return
    }

    validateReplaySchema(replay)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    showWarningToast(`Replay validation: ${message}`, 'error', 'replay-validation-error')
  }
}

export function validateReplayStep(replayStep: any): void {
  try {
    if (typeof replayStep.step !== 'number' || replayStep.step < 0) {
      throw new Error(`Invalid step number: ${replayStep.step}`)
    }

    validateType(replayStep.objects, 'array', 'Replay step objects')

    for (let i = 0; i < replayStep.objects.length; i++) {
      const obj = replayStep.objects[i]
      const prefix = `Step ${replayStep.step}, Object ${i + 1}:`

      if (typeof obj.id !== 'number' || obj.id < 1) {
        throw new Error(`${prefix} Invalid or missing ID`)
      }

      if (typeof obj.type_id !== 'number' || obj.type_id < 0) {
        throw new Error(`${prefix} Invalid type ID`)
      }

      if (obj.location !== undefined) {
        if (!Array.isArray(obj.location) || obj.location.length < 2) {
          throw new Error(`${prefix} Invalid location format`)
        }
        const [x, y] = obj.location
        if (typeof x !== 'number' || typeof y !== 'number') {
          throw new Error(`${prefix} Location coordinates must be numbers`)
        }
      }

      if ('agent_id' in obj) {
        if (typeof obj.agent_id !== 'number' || obj.agent_id < 0) {
          throw new Error(`${prefix} Invalid agent ID`)
        }

        if (obj.action_id !== undefined && (typeof obj.action_id !== 'number' || obj.action_id < 0)) {
          throw new Error(`${prefix} Invalid action ID`)
        }

        if (obj.current_reward !== undefined && typeof obj.current_reward !== 'number') {
          throw new Error(`${prefix} Current reward must be a number`)
        }

        if (obj.total_reward !== undefined && typeof obj.total_reward !== 'number') {
          throw new Error(`${prefix} Total reward must be a number`)
        }
      }

      if (obj.inventory !== undefined) {
        validateType(obj.inventory, 'array', `${prefix} Inventory`)
        for (let j = 0; j < obj.inventory.length; j++) {
          const invItem = obj.inventory[j]
          if (
            !Array.isArray(invItem) ||
            invItem.length !== 2 ||
            typeof invItem[0] !== 'number' ||
            typeof invItem[1] !== 'number'
          ) {
            throw new Error(`${prefix} Invalid inventory item format at index ${j}`)
          }
        }
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    showWarningToast(`Step validation: ${message}`, 'warning', 'step-validation-error')
  }
}
