// Utility functions for converting between different schema formats
// Enhanced with comprehensive Syda constraint support

// All supported Syda field types
export const SYDA_FIELD_TYPES = [
  // Core types
  { value: 'text', label: 'Text', category: 'core' },
  { value: 'string', label: 'String', category: 'core' },
  { value: 'integer', label: 'Integer', category: 'core' },
  { value: 'int', label: 'Int', category: 'core' },
  { value: 'float', label: 'Float', category: 'core' },
  { value: 'number', label: 'Number', category: 'core' },
  { value: 'boolean', label: 'Boolean', category: 'core' },
  { value: 'bool', label: 'Bool', category: 'core' },
  { value: 'date', label: 'Date', category: 'core' },
  { value: 'datetime', label: 'DateTime', category: 'core' },
  { value: 'array', label: 'Array', category: 'core' },
  { value: 'object', label: 'Object', category: 'core' },
  { value: 'foreign_key', label: 'Foreign Key', category: 'core' },
  
  // Extended types
  { value: 'email', label: 'Email', category: 'extended' },
  { value: 'phone', label: 'Phone', category: 'extended' },
  { value: 'address', label: 'Address', category: 'extended' },
  { value: 'url', label: 'URL', category: 'extended' }
] as const

export type SydaFieldType = typeof SYDA_FIELD_TYPES[number]['value']

// Comprehensive Syda constraints interface
export interface SydaConstraints {
  // General constraints
  nullable?: boolean
  unique?: boolean
  primary_key?: boolean
  not_null?: boolean
  
  // Numeric constraints
  min?: number
  max?: number
  decimals?: number
  
  // String constraints
  length?: number
  min_length?: number
  max_length?: number
  pattern?: string
  format?: string
  
  // Date constraints
  min_date?: string
  max_date?: string
  
  // Array constraints
  min_items?: number
  max_items?: number
  item_type?: string
  
  // Enum constraint (for any type)
  enum?: Array<string | number | boolean>
  
  // Foreign key reference
  references?: {
    schema: string
    field: string
  }
}

// Legacy type for backward compatibility
export type FieldType = SydaFieldType

export type SchemaField = {
  id: string
  name: string
  type: SydaFieldType
  description?: string
  constraints?: SydaConstraints
  // Legacy fields for backward compatibility
  required?: boolean
  primaryKey?: boolean
  foreignKey?: { table: string; column: string }
}

export type ParsedSchema = {
  tableName: string
  description: string
  fields: SchemaField[]
}

// Parse YAML content to visual format (simplified parser)
export function parseYamlToVisual(content: string): ParsedSchema {
  const lines = content.split('\n')
  const result: ParsedSchema = {
    tableName: '',
    description: '',
    fields: []
  }
  
  let currentField: Partial<SchemaField> | null = null
  let inConstraints = false
  let inForeignKeys = false
  let foreignKeyMap: Record<string, { table: string; column: string }> = {}
  
  for (const line of lines) {
    const trimmed = line.trim()
    
    if (trimmed.startsWith('__table_name__:')) {
      result.tableName = trimmed.split(':')[1]?.trim() || ''
    } else if (trimmed.startsWith('__description__:')) {
      result.description = trimmed.split(':')[1]?.trim() || ''
    } else if (trimmed.startsWith('__foreign_keys__:')) {
      inForeignKeys = true
      continue
    } else if (inForeignKeys && trimmed.startsWith('  ') && trimmed.includes(':')) {
      // Parse foreign key definition: "  field_name: [Table, column]"
      const match = trimmed.match(/^\s*(\w+):\s*\[(\w+),\s*(\w+)\]/)
      if (match) {
        const [, fieldName, table, column] = match
        foreignKeyMap[fieldName] = { table, column }
      }
    } else if (trimmed && !trimmed.startsWith(' ') && trimmed.includes(':') && !trimmed.startsWith('__')) {
      // New field - save previous field first
      if (currentField && currentField.name) {
        const fk = foreignKeyMap[currentField.name]
        result.fields.push({
          ...currentField,
          foreignKey: fk
        } as SchemaField)
      }
      
      inForeignKeys = false
      inConstraints = false
      currentField = {
        id: Date.now().toString() + Math.random(),
        name: trimmed.split(':')[0].trim(),
        type: 'string',
        required: false,
        primaryKey: false
      }
    } else if (currentField && trimmed.startsWith('type:')) {
      const type = trimmed.split(':')[1]?.trim() as FieldType
      if (type) currentField.type = type
    } else if (currentField && trimmed.startsWith('description:')) {
      currentField.description = trimmed.split(':')[1]?.trim()
    } else if (currentField && trimmed === 'constraints:') {
      inConstraints = true
      currentField.constraints = {}
    } else if (currentField && inConstraints && trimmed.includes('primary_key: true')) {
      currentField.primaryKey = true
    } else if (currentField && inConstraints && trimmed.includes('max_length:')) {
      const maxLength = parseInt(trimmed.split(':')[1]?.trim() || '0')
      if (maxLength > 0) {
        currentField.constraints = { ...currentField.constraints, maxLength }
      }
    }
  }
  
  // Don't forget the last field
  if (currentField && currentField.name) {
    const fk = foreignKeyMap[currentField.name]
    result.fields.push({
      ...currentField,
      foreignKey: fk
    } as SchemaField)
  }
  
  return result
}

// Convert visual format back to YAML (used internally by fieldsToYaml)
function visualToYaml(schema: ParsedSchema): string {
  let yaml = ''
  
  if (schema.tableName) {
    yaml += `__table_name__: ${schema.tableName}\n`
  }
  if (schema.description) {
    yaml += `__description__: ${schema.description}\n`
  }
  
  // Add foreign keys section if any exist (from both legacy and new format)
  const foreignKeyFields = schema.fields.filter(f => f.foreignKey || f.constraints?.references)
  if (foreignKeyFields.length > 0) {
    yaml += `__foreign_keys__:\n`
    foreignKeyFields.forEach(field => {
      if (field.foreignKey) {
        yaml += `  ${field.name}: [${field.foreignKey.table}, ${field.foreignKey.column}]\n`
      } else if (field.constraints?.references) {
        yaml += `  ${field.name}: [${field.constraints.references.schema}, ${field.constraints.references.field}]\n`
      }
    })
  }
  
  yaml += '\n'
  
  // Add fields
  schema.fields.forEach(field => {
    yaml += `${field.name}:\n`
    yaml += `  type: ${field.type}\n`
    
    if (field.description) {
      yaml += `  description: ${field.description}\n`
    }
    
    // Handle both legacy and new constraint formats
    const hasConstraints = field.primaryKey || field.constraints || 
                          (field.constraints && Object.keys(field.constraints).length > 0)
    
    if (hasConstraints) {
      yaml += `  constraints:\n`
      
      // Legacy support
      if (field.primaryKey) {
        yaml += `    primary_key: true\n`
      }
      
      // New Syda constraints
      if (field.constraints) {
        Object.entries(field.constraints).forEach(([key, value]) => {
          if (value !== null && value !== undefined && value !== '') {
            if (typeof value === 'boolean') {
              yaml += `    ${key}: ${value}\n`
            } else if (key === 'enum') {
              // Handle enum as YAML array
              try {
                const enumArray = Array.isArray(value) ? value : JSON.parse(value as string)
                yaml += `    ${key}: [${enumArray.map(v => typeof v === 'string' ? `"${v}"` : v).join(', ')}]\n`
              } catch (e) {
                yaml += `    ${key}: ${JSON.stringify(value)}\n`
              }
            } else if (key === 'references' && typeof value === 'object') {
              yaml += `    ${key}: {schema: "${value.schema}", field: "${value.field}"}\n`
            } else {
              yaml += `    ${key}: ${typeof value === 'string' ? `"${value}"` : value}\n`
            }
          }
        })
      }
    }
  })
  
  return yaml
}

// Convert spreadsheet fields to YAML
export function fieldsToYaml(tableName: string, description: string, fields: any[]): string {
  const schema: ParsedSchema = {
    tableName,
    description,
    fields: fields.map(f => ({
      id: f.id,
      name: f.name,
      type: f.type,
      required: f.required,
      primaryKey: f.primaryKey,
      foreignKey: f.foreignKey,
      description: f.description,
      constraints: f.constraints
    }))
  }
  
  return visualToYaml(schema)
}

// Convert database table to YAML schema
export function databaseTableToYaml(table: any): string {
  let yaml = `__table_name__: ${table.name}\n`
  yaml += `__description__: Imported from database\n`
  
  // Add foreign keys
  const foreignKeyFields = table.columns.filter((col: any) => col.foreignKey)
  if (foreignKeyFields.length > 0) {
    yaml += `__foreign_keys__:\n`
    foreignKeyFields.forEach((col: any) => {
      yaml += `  ${col.name}: [${col.foreignKey.table}, ${col.foreignKey.column}]\n`
    })
  }
  
  yaml += '\n'
  
  // Add columns
  table.columns.forEach((col: any) => {
    yaml += `${col.name}:\n`
    
    // Map database types to our types
    let type = 'string'
    if (col.type.includes('int')) type = 'integer'
    else if (col.type.includes('decimal') || col.type.includes('float') || col.type.includes('double')) type = 'float'
    else if (col.type.includes('bool')) type = 'boolean'
    else if (col.type.includes('date') && col.type.includes('time')) type = 'datetime'
    else if (col.type.includes('date')) type = 'date'
    else if (col.type.includes('text')) type = 'text'
    
    yaml += `  type: ${type}\n`
    
    if (col.primaryKey || !col.nullable) {
      yaml += `  constraints:\n`
      if (col.primaryKey) {
        yaml += `    primary_key: true\n`
      }
    }
  })
  
  return yaml
}

// Validate schema for common issues
export function validateSchema(schema: ParsedSchema): string[] {
  const issues: string[] = []
  
  if (!schema.tableName) {
    issues.push('Table name is required')
  }
  
  if (schema.fields.length === 0) {
    issues.push('At least one field is required')
  }
  
  const primaryKeys = schema.fields.filter(f => f.primaryKey)
  if (primaryKeys.length > 1) {
    issues.push('Multiple primary keys are not supported')
  }
  
  const fieldNames = schema.fields.map(f => f.name)
  const duplicateNames = fieldNames.filter((name, index) => fieldNames.indexOf(name) !== index)
  const uniqueDuplicates = [...new Set(duplicateNames)]
  if (uniqueDuplicates.length > 0) {
    issues.push(`Duplicate field names: ${uniqueDuplicates.join(', ')}`)
  }
  
  // Check foreign key references
  schema.fields.forEach(field => {
    if (field.foreignKey && !field.foreignKey.table) {
      issues.push(`Foreign key for field "${field.name}" is missing table reference`)
    }
  })
  
  return issues
}

// ===== SYDA CONSTRAINT UTILITIES =====

// Get applicable constraints for a field type
export function getApplicableConstraints(fieldType: SydaFieldType): Array<keyof SydaConstraints> {
  const baseConstraints: Array<keyof SydaConstraints> = ['nullable', 'unique', 'primary_key', 'not_null']
  
  switch (fieldType) {
    case 'integer':
    case 'int':
    case 'float':
    case 'number':
      return [...baseConstraints, 'min', 'max', 'decimals', 'enum']
    
    case 'text':
    case 'string':
    case 'email':
    case 'phone':
    case 'address':
    case 'url':
      return [...baseConstraints, 'length', 'min_length', 'max_length', 'pattern', 'format', 'enum']
    
    case 'date':
    case 'datetime':
      return [...baseConstraints, 'format', 'min_date', 'max_date']
    
    case 'array':
      return [...baseConstraints, 'min_items', 'max_items', 'item_type']
    
    case 'foreign_key':
      return [...baseConstraints, 'references']
    
    case 'boolean':
    case 'bool':
      return [...baseConstraints, 'enum']
    
    case 'object':
      return baseConstraints
    
    default:
      return baseConstraints
  }
}

// Get constraint input type for UI
export function getConstraintInputType(constraint: keyof SydaConstraints): 'text' | 'number' | 'checkbox' | 'select' | 'textarea' {
  switch (constraint) {
    case 'nullable':
    case 'unique':
    case 'primary_key':
    case 'not_null':
      return 'checkbox'
    
    case 'min':
    case 'max':
    case 'decimals':
    case 'length':
    case 'min_length':
    case 'max_length':
    case 'min_items':
    case 'max_items':
      return 'number'
    
    case 'pattern':
      return 'textarea'
    
    case 'enum':
      return 'textarea' // JSON array input
    
    case 'references':
      return 'select' // Special handling
    
    default:
      return 'text'
  }
}

// Get constraint label for UI
export function getConstraintLabel(constraint: keyof SydaConstraints): string {
  const labels: Record<keyof SydaConstraints, string> = {
    nullable: 'Nullable',
    unique: 'Unique',
    primary_key: 'Primary Key',
    not_null: 'Not Null',
    min: 'Minimum Value',
    max: 'Maximum Value',
    decimals: 'Decimal Places',
    length: 'Exact Length',
    min_length: 'Minimum Length',
    max_length: 'Maximum Length',
    pattern: 'Regex Pattern',
    format: 'Format',
    min_date: 'Minimum Date',
    max_date: 'Maximum Date',
    min_items: 'Minimum Items',
    max_items: 'Maximum Items',
    item_type: 'Item Type',
    enum: 'Allowed Values',
    references: 'References'
  }
  
  return labels[constraint] || constraint
}

// Get constraint description for UI
export function getConstraintDescription(constraint: keyof SydaConstraints): string {
  const descriptions: Record<keyof SydaConstraints, string> = {
    nullable: 'Allow null/empty values',
    unique: 'Values must be unique across all records',
    primary_key: 'This field is the primary key',
    not_null: 'Field cannot be null or empty',
    min: 'Minimum numeric value allowed',
    max: 'Maximum numeric value allowed',
    decimals: 'Number of decimal places for float values',
    length: 'Exact character length required',
    min_length: 'Minimum character length required',
    max_length: 'Maximum character length allowed',
    pattern: 'Regular expression pattern for validation',
    format: 'Format specification (e.g., "email", "YYYY-MM-DD")',
    min_date: 'Earliest date allowed',
    max_date: 'Latest date allowed',
    min_items: 'Minimum number of items in array',
    max_items: 'Maximum number of items in array',
    item_type: 'Type of items in the array',
    enum: 'List of allowed values (JSON array format)',
    references: 'Foreign key reference to another schema'
  }
  
  return descriptions[constraint] || ''
}

// Validate constraint value
export function validateConstraintValue(constraint: keyof SydaConstraints, value: any, fieldType: SydaFieldType): string | null {
  if (value === null || value === undefined || value === '') {
    return null // Empty values are allowed
  }
  
  switch (constraint) {
    case 'min':
    case 'max':
    case 'decimals':
    case 'length':
    case 'min_length':
    case 'max_length':
    case 'min_items':
    case 'max_items':
      if (isNaN(Number(value)) || Number(value) < 0) {
        return 'Must be a non-negative number'
      }
      break
    
    case 'pattern':
      try {
        new RegExp(value)
      } catch (e) {
        return 'Invalid regular expression pattern'
      }
      break
    
    case 'enum':
      try {
        const parsed = JSON.parse(value)
        if (!Array.isArray(parsed)) {
          return 'Must be a JSON array'
        }
      } catch (e) {
        return 'Must be valid JSON array format'
      }
      break
    
    case 'min_date':
    case 'max_date':
      if (fieldType === 'date' || fieldType === 'datetime') {
        const date = new Date(value)
        if (isNaN(date.getTime())) {
          return 'Must be a valid date'
        }
      }
      break
  }
  
  return null
}

// Convert constraints to YAML format
export function constraintsToYaml(constraints: SydaConstraints): string {
  if (!constraints || Object.keys(constraints).length === 0) {
    return ''
  }
  
  let yaml = '  constraints:\n'
  
  Object.entries(constraints).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      if (typeof value === 'boolean') {
        yaml += `    ${key}: ${value}\n`
      } else if (key === 'enum') {
        // Handle enum as YAML array
        try {
          const enumArray = Array.isArray(value) ? value : JSON.parse(value as string)
          yaml += `    ${key}: [${enumArray.map(v => typeof v === 'string' ? `"${v}"` : v).join(', ')}]\n`
        } catch (e) {
          yaml += `    ${key}: ${JSON.stringify(value)}\n`
        }
      } else if (key === 'references' && typeof value === 'object') {
        yaml += `    ${key}: {schema: "${value.schema}", field: "${value.field}"}\n`
      } else {
        yaml += `    ${key}: ${typeof value === 'string' ? `"${value}"` : value}\n`
      }
    }
  })
  
  return yaml
}
