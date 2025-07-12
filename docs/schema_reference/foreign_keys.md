# Multiple Ways to Define Foreign Keys

Foreign keys can be defined in 2 ways:

1. Using the `__foreign_keys__` special section (recommended):
   ```yaml
   __foreign_keys__:
     user_id: [User, id]
   user_id: foreign_key
   ```

2. Using the field definition with `references`:
   ```yaml
   user_id:
     type: foreign_key
     references:
       schema: User
       field: id
   ```