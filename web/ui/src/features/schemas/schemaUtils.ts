// Utility functions for converting between different schema formats

export type FieldType = 'string' | 'integer' | 'float' | 'boolean' | 'date' | 'datetime' | 'text'

export type SchemaField = {
  id: string
  name: string
  type: FieldType
  required: boolean
  primaryKey: boolean
  foreignKey?: { table: string; column: string }
  description?: string
  constraints?: {
    maxLength?: number
    minValue?: number
    maxValue?: number
  }
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
  
  // Add foreign keys section if any exist
  const foreignKeyFields = schema.fields.filter(f => f.foreignKey)
  if (foreignKeyFields.length > 0) {
    yaml += `__foreign_keys__:\n`
    foreignKeyFields.forEach(field => {
      if (field.foreignKey) {
        yaml += `  ${field.name}: [${field.foreignKey.table}, ${field.foreignKey.column}]\n`
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
    
    if (field.primaryKey || field.constraints) {
      yaml += `  constraints:\n`
      if (field.primaryKey) {
        yaml += `    primary_key: true\n`
      }
      if (field.constraints?.maxLength) {
        yaml += `    max_length: ${field.constraints.maxLength}\n`
      }
      if (field.constraints?.minValue !== undefined) {
        yaml += `    min_value: ${field.constraints.minValue}\n`
      }
      if (field.constraints?.maxValue !== undefined) {
        yaml += `    max_value: ${field.constraints.maxValue}\n`
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
