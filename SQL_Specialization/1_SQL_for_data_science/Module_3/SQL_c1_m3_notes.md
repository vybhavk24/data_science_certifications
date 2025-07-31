# SQL_c1_m3

## Subqueries and Using Sub-Queries

### 1. Quick Refresher

Before diving into subqueries, remember: every subquery is just a `SELECT` statement with its own `FROM`, `WHERE`, and optional aggregates.

Uncorrelated subqueries execute once; correlated subqueries run per row of the outer query.

### 2. Core Explanation

A subquery (or inner query) is a nested `SELECT` placed inside another SQL statement.

It lets you compute a result set or scalar value on the fly to filter, compare, or derive columns without permanently storing intermediate tables.

Types of subqueries:

- Scalar: returns a single value.
- Row: returns a single row of values.
- Table/Derived: used in `FROM` as a temporary table.
- EXISTS/NOT EXISTS: tests presence of rows.

### 3. Syntax & Variants

### Scalar Subquery in SELECT

```sql
SELECT
  p.product_id,
  p.name,
  (SELECT AVG(quantity)
     FROM order_items oi
    WHERE oi.product_id = p.product_id
  ) AS avg_qty_sold
FROM products p;
```

- `(SELECT …)` produces one value per `p.product_id`.
- Correlated: `oi.product_id = p.product_id` links inner to outer.

### Filtering with IN / EXISTS

```sql
-- Using IN
SELECT customer_id, name
  FROM customers
 WHERE customer_id IN (
   SELECT customer_id
     FROM orders
    GROUP BY customer_id
   HAVING SUM(total_amount) > 1000
 );

-- Using EXISTS
SELECT c.customer_id, c.name
  FROM customers c
 WHERE EXISTS (
   SELECT 1
     FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.total_amount > 500
 );
```

- `IN (subquery)`: matches list of values.
- `EXISTS`: checks for at least one matching row; often faster when dealing with large lists.

### Derived Tables in FROM

```sql
SELECT avg_sales.region, avg_sales.avg_amount
  FROM (
    SELECT region, AVG(amount) AS avg_amount
      FROM sales
     GROUP BY region
  ) AS avg_sales
 WHERE avg_sales.avg_amount > 10000;
```

- The inner query acts as a temporary table `avg_sales`.
- You can join this derived table to other tables.

### 4. Real-World Use Cases

- **Threshold Filtering:** Find customers whose lifetime spend exceeds a target using an aggregate subquery.
- **Feature Engineering:** Embed the user’s last login timestamp as a scalar subquery in your user profile extract.
- **Staging & Cleanup:** Use a derived table to identify orphan records before deleting them in a larger `DELETE`statement.
- **Conditional Joins:** Replace complex multi-join logic with an `EXISTS` subquery for performance and clarity.

Compared to joins, subqueries can be more intuitive when you only need a yes/no test or a single aggregated value per row.

### 5. Practice Problems

Assume these tables:

customers

| customer_id | name | signup_date |
| --- | --- | --- |
| 1 | Alice | 2023-01-10 |
| 2 | Bob | 2023-03-22 |
| 3 | Carol | 2023-05-05 |

orders

| order_id | customer_id | total_amount | order_date |
| --- | --- | --- | --- |
| 10 | 1 | 250.0 | 2023-07-01 |
| 11 | 1 | 1250.0 | 2023-08-15 |
| 12 | 2 | 75.0 | 2023-09-05 |
1. Select each customer’s name and their **total spend** using a scalar subquery.
    
    *Hint:* Use `(SELECT SUM(total_amount) FROM orders o WHERE o.customer_id = c.customer_id)`.
    
2. Retrieve customers who have placed **no orders**.
    
    *Hint:* Use `NOT EXISTS` or `NOT IN` with a subquery over `orders`.
    
3. List order IDs and customer names for orders placed **after** the average order date for all orders.
    
    *Hint:* Compute the average date in a scalar subquery, then filter in the outer `WHERE`.
    

### 6. Visual Walkthrough

Given the `customers` and `orders` tables:

1. **Scalar Subquery for Total Spend**
    
    ```sql
    SELECT
      c.name,
      (SELECT SUM(o.total_amount)
         FROM orders o
        WHERE o.customer_id = c.customer_id
      ) AS total_spend
    FROM customers c;
    ```
    
    a. Inner `SUM(o.total_amount)` runs per customer.
    
    b. Outer row returns `name` + computed `total_spend`.
    
2. **EXISTS to Find Non-Ordering Customers**
    
    ```sql
    SELECT name
      FROM customers c
     WHERE NOT EXISTS (
       SELECT 1
         FROM orders o
        WHERE o.customer_id = c.customer_id
     );
    ```
    
    a. For each `c`, `EXISTS` checks `orders`.
    
    b. `NOT EXISTS` filters out customers with any matching orders.
    

This two-step view—compute inner result, then apply in the outer query—illustrates the power and flexibility of subqueries.

---

## Subquery Best Practices and Considerations

### 1. Choose the Right Subquery Form

Subqueries let you nest queries when you need an intermediate result to filter or compute values. They come in scalar, row, table (derived), and existence-check forms. Picking the right form for your scenario helps you write clearer and faster SQL.

- Scalar subqueries return exactly one value per outer row.
- EXISTS/NOT EXISTS check boolean conditions efficiently.
- IN/NOT IN test membership but beware of NULL pitfalls.
- Derived tables (`FROM (… ) AS alias`) stage aggregations or complex filters.

### 2. Performance and Optimization Considerations

Subqueries can hurt performance if used without care. Correlated subqueries re-execute for every outer row, while uncorrelated run once. Understanding how your database optimizer handles each form is key.

- Prefer EXISTS over IN when checking large sets.
- Index join columns to speed up correlated subqueries.
- Replace scalar subqueries in `SELECT` with `JOIN` if they run multiple times.
- Always review the execution plan (`EXPLAIN` or `EXPLAIN ANALYZE`).

### 3. Readability and Maintainability

Deeply nested subqueries become hard to debug and update. Aim for shallow nesting, clear aliases, and self-documenting code so your team can follow your logic months later.

- Use Common Table Expressions (CTEs) to break complex logic into steps.
- Give aliases meaningful names (e.g. `latest_orders` instead of `t1`).
- Comment nonobvious correlations or filters.
- Avoid more than two levels of nesting—refactor into CTEs or views.

### 4. Monitoring and Testing Execution

Testing subqueries against realistic data distributions helps catch performance or correctness issues early. Build small benchmarks and compare alternative strategies.

- Run timing tests on dev or staging with `EXPLAIN ANALYZE`.
- Track key metrics: total cost, buffer reads, execution time.
- Simulate edge cases (no matches, all matches, NULL values).
- Use database-provided profiling tools or logs for long-running queries.

### 5. Summary Comparison

| Subquery Type | Typical Use Case | Correlated? | Performance Note |
| --- | --- | --- | --- |
| Scalar | Inline aggregates per row | Often | Can be slow if not memoized or indexed |
| EXISTS / NOT | Existence checks | Often | Fast with proper indexing |
| IN / NOT IN | Membership against small set | Uncorrelated | Beware NULLs; optimizer may rewrite |
| Derived Tables | Staging, complex aggregations | Uncorrelated | Materialized once; treat as temp table |

---

## Subquery Examples

### 1. Scalar Subquery in SELECT

This pattern computes one aggregated value per row of the outer query without creating a temporary table.

```sql
SELECT
  c.customer_id,
  c.name,
  (
    SELECT COALESCE(SUM(o.total_amount), 0)
      FROM orders o
     WHERE o.customer_id = c.customer_id
  ) AS total_spent
FROM customers c;
```

- The inner `SUM(o.total_amount)` runs once for each customer.
- `COALESCE` ensures customers with no orders show 0 rather than NULL.

### 2. EXISTS Subquery for Filtering

Use `EXISTS` when you only need a yes/no check, and want optimizer-friendly performance on large datasets.

```sql
SELECT
  p.product_id,
  p.name
FROM products p
WHERE EXISTS (
  SELECT 1
    FROM order_items oi
   WHERE oi.product_id = p.product_id
     AND oi.quantity >= 100
);
```

- For each product, the inner query stops as soon as it finds one matching order item.
- This avoids scanning all order items per product.

### 3. IN Subquery for Membership Testing

Great for matching against a small list of values, but watch out if the subquery can return NULLs.

```sql
SELECT
  e.employee_id,
  e.full_name
FROM employees e
WHERE e.department_id IN (
  SELECT d.department_id
    FROM departments d
   WHERE d.location = 'Bangalore'
);
```

- Retrieves employees working in any department located in “Bangalore.”
- If no `location` rows exist, the result is an empty set.

### 4. Derived Table (Table Subquery) for Aggregation & JOIN

Materialize a complex aggregation as a temporary result, then join it back to your main tables.

```sql
SELECT
  ds.region,
  ds.avg_sales,
  r.manager_name
FROM (
  SELECT
    region,
    AVG(amount) AS avg_sales
  FROM sales
  GROUP BY region
) AS ds
JOIN regions r
  ON r.region = ds.region
WHERE ds.avg_sales > 50000;
```

- The inner query (`ds`) calculates average sales by region.
- The outer query filters high-performing regions and pulls in manager info.

### 5. Subquery in UPDATE

Use a scalar subquery to update one table based on an aggregate in another.

```sql
UPDATE users u
SET last_order_date = (
  SELECT MAX(o.order_date)
    FROM orders o
   WHERE o.user_id = u.user_id
)
WHERE EXISTS (
  SELECT 1
    FROM orders o2
   WHERE o2.user_id = u.user_id
);
```

- Sets each user’s `last_order_date` to their most recent purchase.
- The `EXISTS` clause limits updates only to users who have orders.

### 6. DELETE with Subquery

Identify and remove orphan records in a single statement.

```sql
DELETE FROM comments c
 WHERE NOT EXISTS (
   SELECT 1
     FROM posts p
    WHERE p.post_id = c.post_id
 );
```

- Deletes all comments whose `post_id` no longer exists in the `posts` table.
- Ensures referential integrity without extra interim steps.

---

## Joining Tables in SQL

### 1. Quick Refresher

Tables in a relational database link through **keys**.

- A **primary key** uniquely identifies each row in a table.
- A **foreign key** in one table references a primary key in another.

Joins let you combine rows from two or more tables by matching these key columns.

### 2. Core Explanation

A **join** takes every row from one table and pairs it with matching rows in another table based on a condition (usually equality on keys).

Main join types:

- **INNER JOIN**: only matching rows from both tables.
- **LEFT (OUTER) JOIN**: all rows from left table, plus matches or NULLs from right.
- **RIGHT (OUTER) JOIN**: all rows from right table, plus matches or NULLs from left.
- **FULL (OUTER) JOIN**: all rows from both tables, matches where possible, NULLs elsewhere.
- **CROSS JOIN**: Cartesian product—every row from table A with every row from table B.
- **SELF JOIN**: join a table to itself to compare rows (using aliases).
- **NATURAL JOIN**: auto-join on all columns with the same name (use sparingly).

### 3. Syntax & Variants

### Inner Join

```sql
SELECT c.customer_id,
       c.name,
       o.order_id,
       o.total_amount
  FROM customers AS c
  INNER JOIN orders AS o
    ON c.customer_id = o.customer_id;
```

### Left (Outer) Join

```sql
SELECT c.name,
       o.order_id
  FROM customers AS c
  LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id;
```

### Right (Outer) Join

```sql
SELECT e.emp_id,
       d.dept_name
  FROM employees AS e
  RIGHT JOIN departments AS d
    ON e.dept_id = d.dept_id;
```

### Full (Outer) Join

```sql
SELECT a.*, b.*
  FROM table_a AS a
  FULL JOIN table_b AS b
    ON a.key = b.key;
```

### Cross Join

```sql
SELECT *
  FROM regions
  CROSS JOIN quarters;
```

### Self Join

```sql
SELECT m.name      AS manager,
       e.name      AS employee
  FROM employees AS e
  JOIN employees AS m
    ON e.manager_id = m.emp_id;
```

### Natural Join (automatic)

```sql
SELECT *
  FROM products
  NATURAL JOIN categories;
```

### 4. Real-World Use Cases

- **Customer–Order Reports**Inner joins link `customers` + `orders` to show who bought what.
- **Missing Data Analysis**Left joins find customers with zero orders by checking `WHERE o.order_id IS NULL`.
- **Organizational Charts**Self joins map `employees` to their `managers` in the same table.
- **Data Warehousing**Fact tables join to dimension tables (star schema) via foreign keys for BI dashboards.
- **Feature Engineering**Joins combine user activity logs with profile tables to build ML features in a single query.

Compared to subqueries, joins often perform better for multi-table merges and are more readable when linking many tables.

### 5. Practice Problems

Given tables `customers(customer_id, name)` and `orders(order_id, customer_id, total)`:

1. List each customer’s name alongside their order IDs (only customers with orders).
    
    *Hint:* Use `INNER JOIN`.
    
2. Show all customers and their orders; if no order exists, show `NULL` for `order_id`.
    
    *Hint:* Use `LEFT JOIN` and filter or display NULLs.
    
3. Using `employees(emp_id, name, manager_id)`, list every employee and their manager’s name.
    
    *Hint:* Self join `employees` to itself.
    
4. Given `table_a` and `table_b` with some matching and some unique keys, write a query that returns all rows from both tables, indicating which side they came from.
    
    *Hint:* Use `FULL JOIN` and `COALESCE(a.key, b.key)`.
    

### 6. Visual Walkthrough

```tsx
Tables:

Customers                  Orders
┌────────────┐            ┌─────────────┐
│ customer_id│            │ order_id    │
│ name       │            │ customer_id │
│            │            │ total       │
├────────────┤            ├─────────────┤
│ 1, Alice   │            │ 101, 1      │
│ 2, Bob     │            │ 102, 2      │
│ 3, Carol   │            │ 103, 1      │
└────────────┘            └─────────────┘

INNER JOIN result:
Only rows with matching customer_id in both tables.

┌────────────┬───────────┐
│ customer_id│ order_id  │
├────────────┼───────────┤
│     1      │   101     │
│     1      │   103     │
│     2      │   102     │
└────────────┴───────────┘

LEFT JOIN result:
All customers + matching orders (NULL if no match).

┌────────────┬───────────┐
│ customer_id│ order_id  │
├────────────┼───────────┤
│     1      │   101     │
│     1      │   103     │
│     2      │   102     │
│     3      │   NULL    │
└────────────┴───────────┘

FULL JOIN result:
All customers and all orders (NULL where no match).

┌────────────┬───────────┐
│ customer_id│ order_id  │
├────────────┼───────────┤
│     1      │   101     │
│     1      │   103     │
│     2      │   102     │
│     3      │   NULL    │
│    NULL    │   201     │ ← if an order exists with no customer
└────────────┴───────────┘
```

---

## Cartesian Joins, Inner Joins, Aliases & Self Joins

### 1. Quick Refresher

A **join** merges rows from two (or more) tables based on a matching condition—or, in the case of a Cartesian join, *no*condition.

- Tables are sets of rows; joins pair rows across sets.
- A **key** is a column that relates tables.
- **Aliases** are shorthand names for tables (or columns) to simplify queries and support self‐joins.

### 2. Core Explanation

### 2.1 Cartesian Join (CROSS JOIN)

Produces every possible pairing of rows between two tables. If Table A has *m* rows and Table B has *n* rows, the result has *m·n* rows.

Use it when you need all combinations (e.g., generating date/product grids).

### 2.2 Inner Join

Returns only the row pairs where the join condition is true. It’s the most common join: link related records and drop the rest.

### 2.3 Aliases

Short names for tables or columns.

- Simplify long names: `FROM very_long_table_name AS vlt`
- Disambiguate columns when you join a table to itself (self‐join).

### 2.4 Self Join

A table joined to itself to compare rows within the same set. Typically uses two aliases to treat the same table as “left” and “right.”

### 3. Syntax & Variants

### 3.1 Cartesian Join

```sql
-- Explicit CROSS JOIN
SELECT *
  FROM table_a
  CROSS JOIN table_b;

-- Implicit comma syntax (same result)
SELECT *
  FROM table_a, table_b;
```

### 3.2 Inner Join

```sql
SELECT *
  FROM customers AS c
  INNER JOIN orders AS o
    ON c.customer_id = o.customer_id;
```

**Shorthand (implicit INNER JOIN)**

```sql
SELECT *
  FROM customers c, orders o
 WHERE c.customer_id = o.customer_id;
```

### 3.3 Table & Column Aliases

```sql
SELECT
  c.customer_id   AS cid,
  c.name          AS customer_name,
  o.order_id      AS oid,
  o.total_amount  AS amount
FROM customers AS c
JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

### 3.4 Self Join

```sql
SELECT
  e.emp_id        AS employee_id,
  e.name          AS employee_name,
  m.emp_id        AS manager_id,
  m.name          AS manager_name
FROM employees AS e
LEFT JOIN employees AS m
  ON e.manager_id = m.emp_id;
```

### 4. Real‐World Use Cases

- **Cartesian Join**
    - Generate all (user × feature) combinations to initialize feature‐store tables.
    - Create a date × product grid for sales forecasting.
- **Inner Join**
    - Match users to their orders for a purchase history report.
    - Link event logs to user profiles for session analysis.
- **Aliases**
    - Shorten verbose table names in complex queries.
    - Disambiguate columns in multi‐join or self‐join scenarios.
- **Self Join**
    - Build organizational charts: each employee row joins to its manager row.
    - Compare records within the same table (e.g., find overlapping time slots in a schedule).

### 5. Practice Problems

Given these sample tables:

customers

| customer_id | name |
| --- | --- |
| 1 | Alice |
| 2 | Bob |

products

| product_id | product_name |
| --- | --- |
| 1 | Mug |
| 2 | T-shirt |

employees

| emp_id | name | manager_id |
| --- | --- | --- |
| 1 | Alice | NULL |
| 2 | Bob | 1 |
| 3 | Charlie | 1 |

orders

| order_id | customer_id | total |
| --- | --- | --- |
| 101 | 1 | 25.0 |
| 102 | 2 | 40.0 |
1. **Cartesian Join**
    
    Generate every possible *(customer × product)* pair.
    
    *Hint:* Use `CROSS JOIN` between `customers` and `products`.
    
2. **Inner Join**
    
    List each order’s `order_id`, `customer_name` and `total`.
    
    *Hint:* Join `orders` to `customers` on `customer_id`.
    
3. **Aliases**
    
    Rewrite the above inner join using aliases `c` and `o`, and rename `total` to `order_total`.
    
4. **Self Join**
    
    Create a list of each employee alongside their manager’s name.
    
    *Hint:* Self join `employees` with aliases like `e` and `m`.
    

### 6. Visual Walkthrough

```tsx
### 📌 Cartesian Join (CROSS JOIN: customers × products)

Customers                    Products                   Result (CROSS JOIN)
┌──────────────┐            ┌───────────────┐          ┌───────────────┬────────────┐
│ customer_id  │            │ product_id    │          │ customer_id   │ product_id │
│ name         │            │ product_name  │          ├───────────────┼────────────┤
├──────────────┤            ├───────────────┤          │ 1             │ 1          │
│ 1  Alice     │            │ 1  Mug        │          │ 1             │ 2          │
│ 2  Bob       │            │ 2  T-shirt    │          │ 2             │ 1          │
└──────────────┘            └───────────────┘          │ 2             │ 2          │
                                                       └───────────────┴────────────┘

### 📌 Inner Join (orders ↔ customers)

Orders                       Customers                  Result (INNER JOIN)
┌────────────┐              ┌──────────────┐           ┌──────────┬─────────┬───────┐
│ order_id   │              │ customer_id  │           │ order_id │ name    │ total │
│ customer_id│              │ name         │           ├──────────┼─────────┼───────┤
│ total      │              ├──────────────┤           │ 101      │ Alice   │ 25.0  │
├────────────┤              │ 1  Alice     │           │ 102      │ Bob     │ 40.0  │
│ 101, 1, 25 │              │ 2  Bob       │           └──────────┴─────────┴───────┘
│ 102, 2, 40 │              └──────────────┘
└────────────┘

### 📌 Self Join (employees ↔ managers)

Employees (Alias e)                                  Result (SELF JOIN)
┌─────┬─────────┬────────────┐                       ┌─────────────┬─────────────┬────────────┬──────────────┐
│ emp_id │ name    │ manager_id │     →               │ employee_id │ employee_name │ manager_id │ manager_name │
├─────┼─────────┼────────────┤                       ├─────────────┼─────────────┼────────────┼──────────────┤
│ 1     │ Alice   │ NULL       │                     │ 1           │ Alice        │ NULL       │ NULL         │
│ 2     │ Bob     │ 1          │                     │ 2           │ Bob          │ 1          │ Alice        │
│ 3     │ Charlie │ 1          │                     │ 3           │ Charlie      │ 1          │ Alice        │
└─────┴─────────┴────────────┘                       └─────────────┴─────────────┴────────────┴──────────────┘
```

---

## Advanced Joins – Left, Right & Full Outer

### 1. Why These Joins Matter

Left, right, and full outer joins let you merge tables while preserving unmatched rows on one or both sides. They’re essential when your tables don’t share a perfect one-to-one relationship—think sparse labels, optional metadata, or feature completeness in ML pipelines.

### 2. Definitions

- Left Join
    
    Keep every row from the **left** table; bring in matching rows from the right. Unmatched right-side columns become NULL.
    
- Right Join
    
    Keep every row from the **right** table; bring in matching rows from the left. Unmatched left-side columns become NULL.
    
- Full Outer Join
    
    Keep every row from **both** tables. Wherever there’s no match, missing columns are NULL.
    

### 3. Syntax & Examples

Assume two tables, **customers** and **orders**:

customers

| customer_id | name |
| --- | --- |
| 1 | Alice |
| 2 | Bob |
| 3 | Carol |

orders

| order_id | customer_id | total |
| --- | --- | --- |
| 101 | 1 | 25.0 |
| 102 | 2 | 40.0 |

```sql
-- 3.1 LEFT JOIN
SELECT
  c.customer_id,
  c.name,
  o.order_id,
  o.total
FROM customers AS c
LEFT JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

```sql
-- 3.2 RIGHT JOIN
SELECT
  c.customer_id,
  c.name,
  o.order_id,
  o.total
FROM customers AS c
RIGHT JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

```sql
-- 3.3 FULL OUTER JOIN
SELECT
  c.customer_id,
  c.name,
  o.order_id,
  o.total
FROM customers AS c
FULL OUTER JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

### 4. Result Sets Visualized

customers LEFT JOIN orders

| customer_id | name | order_id | total |
| --- | --- | --- | --- |
| 1 | Alice | 101 | 25.0 |
| 2 | Bob | 102 | 40.0 |
| 3 | Carol | NULL | NULL |

customers RIGHT JOIN orders

| customer_id | name | order_id | total |
| --- | --- | --- | --- |
| 1 | Alice | 101 | 25.0 |
| 2 | Bob | 102 | 40.0 |

(same as inner join here—no orders outside customers)

FULL OUTER JOIN

| customer_id | name | order_id | total |
| --- | --- | --- | --- |
| 1 | Alice | 101 | 25.0 |
| 2 | Bob | 102 | 40.0 |
| 3 | Carol | NULL | NULL |

### 5. Real-World Use Cases

- Handling **missing labels**: merge feature table (left) with labels (right); flag unlabeled rows.
- Building **master metadata**: full outer join config (left) with runtime logs (right) to spot missing instrumentation.
- Backfilling **time series**: left join calendar dates to recorded events, filling gaps as NULL.
- Auditing **data integrity**: right join source IDs to target table to catch unprocessed records.

### 6. When & Why to Choose Each

| Join Type | Keeps All Rows From… | Best For |
| --- | --- | --- |
| LEFT JOIN | Left table only | Starting from primary entities, adding related data |
| RIGHT JOIN | Right table only | Rare in practice—reverse of left join |
| FULL OUTER JOIN | Both tables | Comprehensive audit / union of two datasets |

### 7. Practice Problems

Given the tables above plus a new **feedback** table:

feedback

| fb_id | customer_id | rating |
| --- | --- | --- |
| 201 | 2 | 5 |
| 202 | 4 | 3 |
1. Use a **left join** to list every customer and their feedback rating (NULL if none).
2. Use a **right join** to list every feedback entry and the customer name (NULL if unknown).
3. Use a **full outer join** to merge **orders** and **feedback**, showing all interactions.

---

## Unions in SQL: UNION vs. UNION ALL

### 1. Why UNIONS Matter

When you have two (or more) query result sets with the same columns, a UNION lets you stitch them together vertically. This is essential for:

- Consolidating partitions (e.g., regional tables into a global view).
- Combining snapshots taken at different times.
- Merging heterogeneous sources that share a schema.

### 2. UNION vs. UNION ALL: Core Differences

| Feature | UNION | UNION ALL |
| --- | --- | --- |
| Duplicate Removal | Yes (performs a DISTINCT) | No (retains every row) |
| Sort Step | Implicit de-duplication may sort data | No implicit sort or de-dup |
| Performance | Slower on large sets | Faster, minimal overhead |
| Use Case | When you need unique rows | When you want every occurrence |

### 3. Syntax & Examples

Both variants require the same column count and compatible data types, in the same order.

### 3.1 UNION (deduplicated)

```sql
SELECT customer_id, name, total_spend
  FROM customers_us
UNION
SELECT customer_id, name, total_spend
  FROM customers_eu;
```

This returns a single set of unique `(customer_id, name, total_spend)` rows across US and EU customers.

### 3.2 UNION ALL (allow duplicates)

```sql
SELECT customer_id, name, total_spend
  FROM customers_us
UNION ALL
SELECT customer_id, name, total_spend
  FROM customers_eu;
```

This returns every row from both queries, including duplicates if the same customer appears in both regions.

### 4. Performance Considerations

- UNION incurs an implicit DISTINCT operation. The engine must buffer and compare rows, often sorting or hashing them.
- UNION ALL streams rows directly into the output without deduplication overhead.

When your data sets are large and you know there are no overlaps—or duplicates are acceptable—reach for UNION ALL for maximum speed.

### 5. When to Use Which

- Use **UNION** when you need a clean, de-duplicated list. For example, merging user-signup logs across different platforms but reporting each user only once.
- Use **UNION ALL** when you’re building an audit trail, keeping every event, even if it occurred in multiple sources.

### 6. Practical Examples

Given two sales snapshots:

```sql
-- Snapshot on Jan 1
sales_jan AS (
  SELECT order_id, product_id, quantity FROM sales WHERE sale_date = '2025-01-01'
),
-- Snapshot on Jan 2
sales_feb AS (
  SELECT order_id, product_id, quantity FROM sales WHERE sale_date = '2025-02-01'
)
```

1. **Deduplicated historical orders**
    
    ```sql
    SELECT order_id, product_id, quantity
      FROM sales_jan
    UNION
    SELECT order_id, product_id, quantity
      FROM sales_feb;
    ```
    
2. **Full audit of orders**
    
    ```sql
    SELECT order_id, product_id, quantity
      FROM sales_jan
    UNION ALL
    SELECT order_id, product_id, quantity
      FROM sales_feb;
    ```
    

### 7. Practice Problems

1. You have two tables, `completed_tasks_2024` and `completed_tasks_2025`, both with `(task_id, user_id, completed_at)`. Write a query that lists every unique `(user_id, task_id)` across both years.
2. Using the same tables, list every completion event (keep duplicates if a user completed the same task in both years).
3. Combine three monthly reports—`jan_report`, `feb_report`, `mar_report`—into one de-duplicated set of `(item_id, count)`.

---

## SQL and Python: Integrating Databases into Your Python Workflow

### 1. Why Combine SQL and Python

Bringing SQL into Python lets you leverage the database’s power—indexes, set operations, joins, aggregations—while enjoying Python’s flexibility for orchestration, visualization, and ML. You push heavy data work into the database, then pull only the subset you need.

### 2. Core Integration with Python’s DB-API

All major RDBMS drivers in Python implement [PEP 249](https://www.python.org/dev/peps/pep-0249/), the DB-API specification. The flow is:

1. Import the driver (e.g., `import psycopg2` or `import sqlite3`).
2. Open a connection: `conn = psycopg2.connect(...)`.
3. Create a cursor: `cur = conn.cursor()`.
4. Execute SQL: `cur.execute(sql, params)`.
5. Fetch results: `rows = cur.fetchall()`.
6. Close cursor and connection.

Using context managers (`with conn, conn.cursor()`) ensures proper cleanup even on errors.

### 3. Key Libraries Comparison

| Library | Use Case | Code Style |
| --- | --- | --- |
| psycopg2 / pyodbc | Raw SQL against Postgres / SQL Server | DB-API cursor calls |
| sqlite3 | Built-in, lightweight, ideal for prototyping | DB-API cursor calls |
| SQLAlchemy Core | Build SQL expressions programmatically | Fluent Python objects |
| SQLAlchemy ORM | Map tables to Python classes (CRUD) | Class definitions & sessions |
| pandas.read_sql | Query directly into DataFrames | One-line queries |
| asyncpg | Async Postgres for high-throughput scenarios | `async/await` calls |

### 4. Raw SQL Example with psycopg2

```python
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# 1. Load credentials securely
DB_URL = os.getenv("DATABASE_URL")

# 2. Connect and create cursor
with psycopg2.connect(DB_URL) as conn:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 3. Parameterized query to avoid SQL injection
        cur.execute(
            "SELECT user_id, email FROM users WHERE signup_date >= %s;",
            ("2025-01-01",)
        )
        users = cur.fetchall()  # List of dicts: [{"user_id":1, "email":"..."}]
print(users)
```

- **RealDictCursor** returns each row as a Python dict.
- Parameter substitution (`%s`) is driver-safe.

### 5. SQLAlchemy Core & ORM

### Core (Expression Language)

```python
from sqlalchemy import create_engine, MetaData, Table, select, func

engine = create_engine("postgresql://user:pass@host/dbname")
metadata = MetaData(bind=engine)

users = Table("users", metadata, autoload_with=engine)
orders = Table("orders", metadata, autoload_with=engine)

# Build a JOIN + aggregate in pure Python objects
stmt = (
    select(users.c.user_id, func.count(orders.c.order_id).label("order_count"))
    .join(orders, users.c.user_id == orders.c.user_id)
    .group_by(users.c.user_id)
)

with engine.connect() as conn:
    result = conn.execute(stmt).fetchall()
print(result)
```

### ORM (Object-Relational Mapping)

```python
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True)
    email   = Column(String, nullable=False)
    orders  = relationship("Order", back_populates="user")

class Order(Base):
    __tablename__ = "orders"
    order_id = Column(Integer, primary_key=True)
    user_id  = Column(Integer, ForeignKey("users.user_id"))
    user     = relationship("User", back_populates="orders")

Session = sessionmaker(bind=engine)
with Session() as session:
    # Query via ORM
    active_users = (
        session.query(User)
        .filter(User.email.ilike("%@example.com"))
        .all()
    )
    for user in active_users:
        print(user.email, len(user.orders))
```

### 6. pandas for SQL Integration

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///data.db")

# Read filtered data into DataFrame
df = pd.read_sql_query(
    "SELECT * FROM sales WHERE sale_date >= :start_date;",
    con=engine,
    params={"start_date": "2025-01-01"},
    parse_dates=["sale_date"]
)

# Chunked reads for large tables
iter_df = pd.read_sql_query(
    "SELECT * FROM events;",
    con=engine,
    chunksize=100000
)
for chunk in iter_df:
    process(chunk)
```

- `read_sql_query` and `read_sql_table` bridge SQL and DataFrames seamlessly.
- Use `chunksize` to process big tables in batches.

### 7. Best Practices

- **Parameterize** all queries to prevent injection.
- **Pool connections** using SQLAlchemy or `psycopg2.pool` for high-throughput apps.
- **Store credentials** in environment variables or secret stores, never in code.
- **Use CTEs** (Common Table Expressions) for complex logic, then call them from Python.
- **Profile queries** with `EXPLAIN ANALYZE` before embedding them in pipelines.

### 8. Real-World Use Cases

1. **ETL Pipelines**: Python orchestrates Airflow tasks that extract via SQL, transform in pandas, and load back.
2. **Feature Stores**: SQL builds aggregated features; Python collects them into ML datasets.
3. **Reporting APIs**: FastAPI endpoints execute parameterized SQL and return JSON.
4. **Ad Hoc Analysis**: Jupyter notebooks let you prototype SQL queries, visualize with `matplotlib` or `seaborn`, and iterate quickly.

### 9. Practice Problems

1. Connect to a local SQLite database, create a `products` table, insert sample rows, then query all products priced over $20 into a pandas DataFrame.
2. Using SQLAlchemy Core, write a query that finds the top 3 customers by total spend, then fetch results.
3. Define ORM models for `Employee` and `Department`. Query employees whose department name starts with “Sales”.
4. In psycopg2, write a transaction that moves $50 from account A to B, with proper `BEGIN`/`COMMIT` and rollback on error.

### 10. Visual Workflow Diagram

```
[ Python Script / Notebook ]
           │
           ├─> create_engine / connect()
           │
           ├─> execute SQL (raw / SQLAlchemy / pandas)
           │
     [ Database Server ]
           │
      retrieves data
           │
   returns rows → Python
           │
           └─> process in DataFrame / objects / plots
```

---