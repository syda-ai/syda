## Structured Data Example

```yaml
__table_name__: Order
__table_description__: Orders for an e-commerce system
__depends_on__: [Customer, Product]
__foreign_keys__:
  customer_id: [Customer, id]

id:
  type: integer
  description: Order ID
  constraints:
    primary_key: true
    not_null: true

customer_id:
  type: integer
  description: Customer reference
  constraints:
    not_null: true

order_date:
  type: date
  constraints:
    format: YYYY-MM-DD
    not_null: true

total_amount:
  type: float
  description: Total order amount
  constraints:
    min: 0
    decimals: 2

status:
  type: text
  description: Order processing status
  constraints:
    enum: ["pending", "processing", "shipped", "delivered", "cancelled"]

items:
  type: array
  description: Line items in the order
```


## Template Schema Example

```yaml
__template__: true
__table_description__: Invoice document template
__name__: Invoice
__depends_on__: [Customer, Order, OrderItem]
__foreign_keys__:
  customer_id: [Customer, id]
  order_id: [Order, id]

__template_source__: /path/to/invoice_template.html
__input_file_type__: html
__output_file_type__: pdf

invoice_number:
  type: string
  pattern: 'INV-\d{6}'
  description: Unique invoice identifier

customer_id:
  type: integer
  description: Reference to customer

order_id:
  type: integer
  description: Reference to order

issue_date:
  type: date
  format: YYYY-MM-DD
  description: Date when invoice was issued

due_date:
  type: date
  format: YYYY-MM-DD
  description: Payment due date

items:
  type: array
  description: Line items from the order
```