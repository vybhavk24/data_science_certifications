# SQL_c1_m1

## What Is SQL?

### 1. Introduction

SQL (Structured Query Language) is the standard language for interacting with relational databases. It lets you:

- Define and modify database structures (tables, schemas).
- Insert, update, delete, and retrieve data.
- Express *what* you want (e.g., “give me all users over 30”) without detailing *how* the database engine fulfills it.

At its heart, SQL is **declarative**—you describe the desired result, and the database figures out the most efficient execution plan.

### 2. Simple Syntax & Variants

Here’s the most common SQL statement, `SELECT`. It illustrates SQL’s basic structure:

```sql
SELECT column1, column2
  FROM table_name
 WHERE condition
 ORDER BY column1 DESC;
```

Breakdown:

- `SELECT column1, column2`Lists the fields you want back.
- `FROM table_name`Specifies which table holds the data.
- `WHERE condition`Filters rows to only those matching your criteria.
- `ORDER BY column1 DESC`Sorts the output by `column1` in descending order.

Variants & shorthand:

- `SELECT * FROM table_name;`Retrieves all columns.
- Omit `ORDER BY` for unsorted results.
- Use comparison operators in `WHERE`: `=`, `>`, `<`, `IN`, `LIKE`, etc.

### 3. Real-World Use Cases

- **Data Analysis & Dashboards**You might write a `SELECT` with `GROUP BY` and aggregates (`SUM`, `COUNT`) to power a sales dashboard.
- **Business Reporting**Filtering and joining tables (e.g., customers + orders) underlies monthly revenue reports.
- **Machine Learning Prep**SQL helps you pull and clean feature tables—dropping nulls, aggregating event logs—before feeding them into an ML pipeline.

Comparisons:

- Declarative vs. procedural scripts (e.g., Python loops). SQL lets the database optimize queries, often running faster on large data.

### 4. Practice Problems

Given the table `employees`:

| id | name | department | salary |
| --- | --- | --- | --- |
| 1 | Alice | HR | 70000 |
| 2 | Bob | Engineering | 95000 |
| 3 | Charlie | HR | 65000 |
| 4 | Denise | Marketing | 72000 |
1. Retrieve all columns for employees in the HR department.
    
    *Hint:* Filter on `department`.
    
2. List only `name` and `salary`, sorted by `salary` descending.
    
    *Hint:* Use `ORDER BY salary DESC`.
    
3. Count how many employees work in each department.
    
    *Hint:* Group by `department` and use `COUNT(*)`.
    

### 5. Visual Walkthrough

Initial `employees` table:

| id | name | department | salary |
| --- | --- | --- | --- |
| 1 | Alice | HR | 70000 |
| 2 | Bob | Engineering | 95000 |
| 3 | Charlie | HR | 65000 |
| 4 | Denise | Marketing | 72000 |

Example:

```sql
SELECT name, salary
  FROM employees
 WHERE department = 'HR'
 ORDER BY salary DESC;
```

Step-by-step transformation:

1. **FROM & WHERE**
    
    Filters to HR rows:
    
    | id | name | department | salary |
    | --- | --- | --- | --- |
    | 1 | Alice | HR | 70000 |
    | 3 | Charlie | HR | 65000 |
2. **SELECT**
    
    Projects only `name` and `salary`:
    
    | name | salary |
    | --- | --- |
    | Alice | 70000 |
    | Charlie | 65000 |
3. **ORDER BY**
    
    Sorts by `salary` descending:
    
    | name | salary |
    | --- | --- |
    | Alice | 70000 |
    | Charlie | 65000 |

---

## Data Models – Thinking About Your Data

### 1. Quick Refresher

A data model defines how you organize information into tables, columns, and relationships.

Every table represents an entity (for example, `users` or `orders`).

Columns hold attributes (like `user_id`, `email`, `order_date`).

Keys—primary and foreign—connect tables and enforce uniqueness and referential integrity.

### 2. Core Explanation

A solid data model ensures queries run fast, data stays accurate, and your analyses make sense. There are three main layers:

| Layer | Purpose |
| --- | --- |
| Conceptual | High-level entities and relationships |
| Logical | Detailed tables, columns, and keys (no SQL yet) |
| Physical | Actual SQL tables, data types, indexes, partitions |

Building a model starts with mapping real-world objects to tables, defining attributes, and sketching how tables link via keys.

### 3. Syntax & Variants

### CREATE TABLE with constraints

```sql
CREATE TABLE customers (
  customer_id   INT          PRIMARY KEY,
  first_name    VARCHAR(50)  NOT NULL,
  last_name     VARCHAR(50)  NOT NULL,
  email         VARCHAR(100) UNIQUE
);

CREATE TABLE orders (
  order_id      INT          PRIMARY KEY,
  customer_id   INT          NOT NULL,
  order_date    DATE         NOT NULL,
  total_amount  DECIMAL(10,2),
  FOREIGN KEY (customer_id)
    REFERENCES customers(customer_id)
);
```

Breakdown of clauses:

- `INT`, `VARCHAR`, `DECIMAL` define data types.
- `PRIMARY KEY` enforces one unique identifier per row.
- `NOT NULL` forbids missing values.
- `UNIQUE` prevents duplicate entries in a column.
- `FOREIGN KEY` links tables, ensuring referential integrity.

Variants & best practices:

- Use surrogate keys (`SERIAL` or auto-increment) vs. natural keys (`email`).
- Choose `VARCHAR` length based on maximum expected input.
- Normalize to 3NF (no repeating groups, dependencies only on keys).

### 4. Real-World Use Cases

- Analytics DashboardsStar schemas place fact tables (`sales`) at the center, joined to dimension tables (`date`, `product`, `store`).
- ETL & Data WarehousingLogical modeling shows source tables; physical design adds partitions and indexes for faster loads.
- Machine Learning Feature StoresWell-modeled tables let you join user actions, demographics, and labels without redundancy or missing links.

Comparison:

- Flat table vs. normalized modelFlat tables are easy to query but waste space and risk inconsistencies.Normalized models avoid duplication but may require more joins.

### 5. Practice Problems

Given a blogging platform, design tables for users, posts, and comments.

1. Identify entities and their key attributes.
2. Sketch primary/foreign keys between tables.
3. Write `CREATE TABLE` statements with appropriate constraints.

Hints:

- A comment belongs to one post and one user.
- A post belongs to one user.
- Emails must be unique.

### 6. Visual Walkthrough

Conceptual view:

[Users]──(writes)──[Posts]──(has)──[Comments]

Logical tables and keys:

| Table | Primary Key | Foreign Key |
| --- | --- | --- |
| users | user_id | — |
| posts | post_id | user_id → users |
| comments | comment_id | post_id → posts |
|  |  | user_id → users |

ASCII ER diagram:

```
users                   posts                   comments
+------------+          +-----------+           +-------------+
| user_id PK |◄─────────| user_id FK|           | comment_id PK|
| email  UQ  |          | post_id PK|◄──────┐   | post_id FK   |
| name       |          | title     |       │   | user_id FK   |
+------------+          +-----------+       │   | content      |
                                             └──►+-------------+
```

Rows travel:

1. A new user row enters `users`.
2. That user’s posts appear in `posts`, linking by `user_id`.
3. Their comments join both `posts` and `users`, ensuring traceability.

---

## Data Models – Evolution of Data Models

### 1. Quick Refresher

A data model is the blueprint that defines how data is stored, organized, and related.

- Tables, columns, and keys in relational models.
- Schemas or collections in document stores.
- Nodes and edges in graph databases.

Each model arises to solve specific needs around volume, complexity, and performance.

### 2. Core Explanation

As data volumes, variety, and application demands grew, the one-size-fits-all relational model faced limits. New paradigms emerged:

| Era | Model | Key Drivers |
| --- | --- | --- |
| 1960s–70s | Hierarchical & Network | Early hardware constraints, navigation APIs |
| 1980s | Relational | Declarative queries, normalization, portability |
| 1990s | Object-Relational | Complex data types, OOP alignment |
| 2000s | Dimensional (Star) | BI reporting, OLAP cubes |
| 2010s | NoSQL (Key-Value, Document, Column) | Big data scale, schema flexibility |
| 2010s+ | Graph | Highly connected data (social, recommendation) |
| 2020s+ | Multi-Model & Lakehouse | Unified analytics across diverse stores |

Each step responded to new application patterns and hardware advances.

### 3. Syntax & Variants

### Relational (Normalized)

```sql
CREATE TABLE customer (
  customer_id INT PRIMARY KEY,
  name        VARCHAR(100),
  email       VARCHAR(100) UNIQUE
);

CREATE TABLE order (
  order_id    INT PRIMARY KEY,
  customer_id INT REFERENCES customer(customer_id),
  order_date  DATE
);
```

### Dimensional (Star Schema)

```sql
CREATE TABLE fact_sales (
  sale_id     INT PRIMARY KEY,
  date_id     INT,
  product_id  INT,
  store_id    INT,
  amount      DECIMAL(10,2)
);

CREATE TABLE dim_date (
  date_id     INT PRIMARY KEY,
  date        DATE,
  month       INT,
  year        INT
);
```

### Document (e.g., MongoDB)

```jsx
db.users.insertOne({
  _id: ObjectId(),
  name: "Alice",
  purchases: [
    { order_id: 1, amount: 95.50 },
    { order_id: 2, amount: 42.10 }
  ]
});
```

### Key-Value (e.g., Redis)

```
SET session:abcd1234 "{ \"user_id\": 42, \"expires\": \"2025-08-01\" }"
GET session:abcd1234
```

### Graph (e.g., Cypher for Neo4j)

```
CREATE (u:User {id: 1, name: "Bob"})
CREATE (p:Product {id: 100, name: "Laptop"})
CREATE (u)-[:BOUGHT {date:"2025-07-29"}]->(p)
```

### 4. Real-World Use Cases

- Relational: banking systems with strict ACID guarantees.
- Dimensional: enterprise data warehouses powering dashboards.
- Document: content management and user profiles with varying schemas.
- Key-Value: session storage, caching, feature flags in web apps.
- Column-Family: time-series and IoT data at massive scale.
- Graph: social networks, fraud detection, recommendation engines.
- Multi-Model/Lakehouse: teams querying structured tables and semi-structured logs in one place.

Comparisons help you pick the right fit for performance, scalability, and schema flexibility.

### 5. Practice Problems

1. Choose a model for a ride-sharing app (drivers, riders, trips, surge zones).
    
    Hint: Highly connected—think graph vs. relational.
    
2. Design a star schema for an online retailer tracking sales by product, customer, and region.
    
    Hint: Identify facts vs. dimensions.
    
3. Show how you’d store user settings (theme, notifications) in a key-value store.
    
    Hint: Key naming conventions and JSON values.
    

## 6. Visual Walkthrough

Hierarchical vs. Relational vs. Star vs. Graph:

```
Hierarchical                Relational                    Star Schema

     Company                     users                  fact_sales ── dim_date
     /     \                     ┌───────┐               ┌───────┐   ┌──────┐
  Dept A  Dept B  vs.           │user_id│  vs.          │date_id│   │ date │
   |         |                  │ name  │               │amount │   └──────┘
 Employees                employees─┬─orders       date_dim─┬─product_dim
 ┌─────┐                     ┌──────┐    ┌──────┐            └─store_dim
 │emp1 │                     │emp1  │    │ord1  │
 └─────┘                     └──────┘    └──────┘
```

Graph model example:

```
(User)-[:FRIENDS_WITH]->(User)
   \
    [:BOUGHT]->(Product)
```

This evolution shows how each model’s structure aligns with its problem domain.

---

## Relational vs Transactional Models

### 1. Quick Refresher

The **relational model** is the theory behind structuring data into tables (relations) with rows and columns, using keys to represent relationships.

A **transaction** is a unit of work—one or more SQL statements—that must either fully succeed or fully fail, preserving data integrity.

### 2. Core Explanation

**Relational Model**

- Defines *how* you organize data: tables, columns, primary/foreign keys, and normalization rules.
- Focuses on data structure and logical consistency.

**Transactional Model**

- Defines *how* you execute operations on your relational data in a safe, atomic way.
- Enforces ACID properties:
    - Atomicity: all or nothing
    - Consistency: valid state transitions
    - Isolation: concurrent transactions don’t interfere
    - Durability: once committed, changes persist

Think of the relational model as the blueprint of your database and the transactional model as the guardrails for every change you make.

### 3. Syntax & Variants

### Defining Tables (Relational Model)

```sql
CREATE TABLE accounts (
  account_id   SERIAL      PRIMARY KEY,
  owner_name   VARCHAR(100) NOT NULL,
  balance      DECIMAL(12,2) DEFAULT 0.00
);
```

- `SERIAL`: auto-incrementing integer.
- `PRIMARY KEY`: unique identifier.
- `NOT NULL`: disallows missing values.
- `DEFAULT`: starting value if none provided.

### Controlling Transactions (Transactional Model)

```sql
-- Start a transaction
BEGIN;

-- Update two related tables
UPDATE accounts
   SET balance = balance - 250.00
 WHERE account_id = 101;

UPDATE accounts
   SET balance = balance + 250.00
 WHERE account_id = 202;

-- Finalize or cancel
COMMIT;
-- or if something went wrong:
-- ROLLBACK;
```

Variants & best practices:

- Implicit transactions (some clients auto-commit each statement).
- Explicit `BEGIN ... COMMIT` for multi-step logic.
- Use savepoints for partial rollbacks:
    
    ```sql
    SAVEPOINT transfer_midway;
    -- later:
    ROLLBACK TO SAVEPOINT transfer_midway;
    ```
    

### 4. Real-World Use Cases

- **Banking & Finance**Money transfers must debit one account and credit another *atomically*.
- **E-Commerce**Reserving stock and creating an order must succeed together to avoid overselling.
- **Event Sourcing / Audit Logs**You record events only once a transaction commits to guarantee data lineage.

**Why This Matters for Data Science**

- Guarantees your feature tables aren’t built on half-applied updates.
- Enables reproducible reports: if a pipeline fails mid-run, you won’t mix old and new data.

### 5. Practice Problems

Given `accounts(account_id, owner_name, balance)`:

1. Write a transaction that transfers $100 from account 1 to account 2, rolling back if either update fails.
    
    *Hint:* Wrap both `UPDATE` statements between `BEGIN` and `COMMIT`.
    
2. Insert a new account and immediately set a savepoint. Then deliberately cause an error on a second insert, roll back to the savepoint, and commit.
    
    *Hint:* Use `SAVEPOINT` and `ROLLBACK TO`.
    
3. Simulate a reporting table update:
    - Create `daily_balance(account_id, report_date, balance)`.
    - In one transaction, delete yesterday’s rows for a given date and insert fresh ones.

### 6. Visual Walkthrough

Initial `accounts` table:

| account_id | owner_name | balance |
| --- | --- | --- |
| 101 | Alice | 1,200.00 |
| 202 | Bob | 800.00 |

Transaction flow for $250 transfer:

```
BEGIN
  └─ Read balance(101)=1200.00
  └─ Read balance(202)=800.00

  UPDATE 101: 1200.00 − 250.00 → 950.00
  UPDATE 202:  800.00 + 250.00 →1050.00
COMMIT
```

After `COMMIT`:

| account_id | owner_name | balance |
| --- | --- | --- |
| 101 | Alice | 950.00 |
| 202 | Bob | 1,050.00 |

If any step fails before `COMMIT`, a `ROLLBACK` restores the original balances, preserving atomicity.

---

## Retrieving Data with SELECT Statements

### 1. Quick Refresher

Tables are collections of rows (records) and columns (fields).

Each row is one entity instance; each column is an attribute.

SELECT pulls and reshapes those rows and columns into your result set.

### 2. Core Explanation

SELECT is the most fundamental SQL command. It declares *what* data you want, and lets the database engine optimize *how* to fetch it.

Basic flow of a SELECT:

- Identify source tables (`FROM`)
- Optionally filter rows (`WHERE`)
- Choose which columns or expressions to project (`SELECT`)
- Optionally reorder (`ORDER BY`) or page results (`LIMIT`/`OFFSET`)

Because it’s declarative, you don’t write loops—the engine figures out the fastest plan.

### 3. Syntax & Variants

### Basic SELECT

```sql
SELECT column1, column2
  FROM table_name;
```

Breakdown:

- `SELECT column1, column2`: list of columns or expressions.
- `FROM table_name`: source table or view.

### Common Variants

- Select all columns:
    
    ```sql
    SELECT * FROM employees;
    ```
    
- Filter rows:
    
    ```sql
    SELECT id, name
      FROM employees
     WHERE department = 'Engineering';
    ```
    
- Rename columns (alias):
    
    ```sql
    SELECT
      first_name AS fname,
      last_name  AS lname
      FROM users;
    ```
    
- Remove duplicates:
    
    ```sql
    SELECT DISTINCT department
      FROM employees;
    ```
    
- Sort results:
    
    ```sql
    SELECT id, salary
      FROM employees
     ORDER BY salary DESC;
    ```
    
- Limit and paginate (PostgreSQL/MySQL):
    
    ```sql
    SELECT *
      FROM orders
     ORDER BY order_date DESC
     LIMIT 10
    OFFSET 20;
    ```
    
- Top-N style (SQL Server):
    
    ```sql
    SELECT TOP 5 *
      FROM sales
     ORDER BY amount DESC;
    ```
    

### 4. Real-World Use Cases

- **Data Exploration**Analysts run quick SELECT * queries with `WHERE` filters to inspect data quality.
- **Feature Extraction**Data scientists pull specific columns and apply expressions (e.g., `price * quantity`) before modeling.
- **Report Generation**Business intelligence tools embed SELECT statements to feed dashboards with filtered and sorted data.
- **Pagination in Web Apps**APIs use `LIMIT`/`OFFSET` to return page-by-page results for user interfaces.

Comparisons:

- SELECT vs. procedural loops in Python: SELECT pushes computation to the database, reducing data transfer and speeding up large-scale operations.

### 5. Practice Problems

Assume a table `employees`:

| id | first_name | last_name | department | salary |
| --- | --- | --- | --- | --- |
| 1 | Alice | Johnson | Engineering | 95000 |
| 2 | Bob | Smith | Marketing | 72000 |
| 3 | Carla | Gomez | Engineering | 102000 |
| 4 | David | Lee | Sales | 68000 |
1. List all columns for employees in “Engineering”.
    
    Hint: filter with `WHERE department = 'Engineering'`.
    
2. Retrieve each unique department in the company.
    
    Hint: use `SELECT DISTINCT`.
    
3. Show the top 2 highest-paid employees (all columns).
    
    Hint: sort by `salary DESC` and apply a limit.
    

### 6. Visual Walkthrough

Starting table:

| id | first_name | last_name | department | salary |
| --- | --- | --- | --- | --- |
| 1 | Alice | Johnson | Engineering | 95000 |
| 2 | Bob | Smith | Marketing | 72000 |
| 3 | Carla | Gomez | Engineering | 102000 |
| 4 | David | Lee | Sales | 68000 |

Example query:

```sql
SELECT id, first_name, salary
  FROM employees
 WHERE department = 'Engineering'
 ORDER BY salary DESC
 LIMIT 2;
```

Step-by-step:

1. FROM & WHERE → filter to Engineering:
    
    
    | id | first_name | salary |
    | --- | --- | --- |
    | 1 | Alice | 95000 |
    | 3 | Carla | 102000 |
2. SELECT → project only needed columns:
    
    
    | id | first_name | salary |
    | --- | --- | --- |
    | 1 | Alice | 95000 |
    | 3 | Carla | 102000 |
3. ORDER BY → sort descending:
    
    
    | id | first_name | salary |
    | --- | --- | --- |
    | 3 | Carla | 102000 |
    | 1 | Alice | 95000 |
4. LIMIT → take top 2 (already two rows):

Final result:

| id | first_name | salary |
| --- | --- | --- |
| 3 | Carla | 102000 |
| 1 | Alice | 95000 |

---

## Creating Tables

### 1. Quick Refresher

A table is the fundamental container in a relational database.

- Each table represents one entity (for example, `customers` or `orders`).
- Columns define attributes and data types (like `INT`, `VARCHAR`, `DATE`).
- Rows store individual records.

Before you can insert or query data, you must **define the table’s structure** with `CREATE TABLE`.

### 2. Core Explanation

`CREATE TABLE` is a Data Definition Language (DDL) command that tells the database:

- What columns exist
- Which data type each column holds
- Any rules or constraints (e.g., unique keys, default values)

Under the hood, the database allocates storage structures and updates its metadata so it knows how to store and validate your data.

### 3. Syntax & Variants

### Basic Syntax

```sql
CREATE TABLE table_name (
  column1   DATA_TYPE  [column_constraint],
  column2   DATA_TYPE  [column_constraint],
  ...
  [table_constraint]
);
```

- `table_name`: your new table’s identifier.
- `DATA_TYPE`: e.g., `INT`, `VARCHAR(50)`, `DATE`, `DECIMAL(10,2)`.
- `column_constraint`: `NOT NULL`, `UNIQUE`, `PRIMARY KEY`, `DEFAULT`, `CHECK(...)`.
- `table_constraint`: multi-column constraints like `PRIMARY KEY(col1, col2)` or `FOREIGN KEY(colX) REFERENCES other_table(colY)`.

### Common Variants

- **IF NOT EXISTS**
    
    ```sql
    CREATE TABLE IF NOT EXISTS customers ( … );
    ```
    
    Skips creation if the table already exists.
    
- **Temporary Tables** (session-scoped)
    
    ```sql
    CREATE TEMPORARY TABLE temp_orders ( … );
    ```
    
    Automatically drops when your session ends.
    
- **Partitioned Tables** (PostgreSQL example)
    
    ```sql
    CREATE TABLE logs (
      log_id   BIGSERIAL,
      log_date DATE       NOT NULL,
      message  TEXT
    ) PARTITION BY RANGE (log_date);
    ```
    

### 4. Real-World Use Cases

- **OLTP Systems**
    
    Define core tables (`users`, `products`, `transactions`) with strict constraints for data integrity in banking or e-commerce.
    
- **Data Warehousing**
    
    Create fact tables (`fact_sales`) and dimension tables (`dim_date`, `dim_customer`) using partitioning and distribution keys for high-performance analytics.
    
- **ETL & Staging**
    
    Spin up temporary tables to load raw CSV or JSON data, clean it, then insert into production tables.
    
- **Feature Stores for ML**
    
    Define tables to hold aggregated features (`user_id`, `avg_session_time`, `purchase_count`) with defaults and update strategies.
    

### 5. Practice Problems

1. **User Management Table**
    
    Define a `users` table with:
    
    - `user_id` as auto-incrementing primary key
    - `email` as unique, non-null text
    - `created_at` defaulting to the current timestamp
    
    *Hint:* Use `SERIAL` or `BIGSERIAL` for the PK and `DEFAULT CURRENT_TIMESTAMP`.
    
2. **Product Inventory Table**
    
    Create `inventory` with:
    
    - `product_id` (PK)
    - `warehouse_id` (PK) — composite key with `product_id`
    - `quantity` integer, non-negative
    - Constraint to ensure `quantity >= 0`
    
    *Hint:* Use a table-level `PRIMARY KEY(product_id, warehouse_id)` and a `CHECK (quantity >= 0)`.
    
3. **Daily Metrics Temp Table**
    
    In one SQL snippet, create a temporary table `daily_metrics` with columns `metric_date DATE` and `value NUMERIC(12,2)`.
    
    *Hint:* Prefix with `CREATE TEMPORARY TABLE`.
    

### 6. Visual Walkthrough

1. **Before Creation**
    
    No `users` table exists in your schema.
    
2. **DDL Command**
    
    ```sql
    CREATE TABLE users (
      user_id    SERIAL       PRIMARY KEY,
      email      VARCHAR(255) NOT NULL UNIQUE,
      created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    ```
    
3. **Database Perspective**
    
    ```tsx
    Database Perspective
    ┌──────────────────────────┐
    │       Schema: public     │
    │  ┌──────────────────────┐│
    │  │     Table: users     ││
    │  │      Columns:        ││
    │  │  • user_id SERIAL     │
    │  │  • email VARCHAR(255) │
    │  │  • created_at TIMEST  │
    │  └──────────────────────┘│
    └──────────────────────────┘
    ```
    
4. **After Inserting Data**
    
    
    | user_id | email | created_at |
    | --- | --- | --- |
    | 1 | alice@example.com | 2025-07-29 23:30:00 |
    | 2 | bob@example.com | 2025-07-29 23:35:00 |

The table now enforces uniqueness on `email`, auto-numbers `user_id`, and timestamps new rows automatically.

---

## Creating Temporary Tables

### 1. Quick Refresher

A **regular table** persists in the database schema until you drop it.

Temporary tables live only for the duration of your session (or transaction), helping you stage, transform, or aggregate data without cluttering your schema.

### 2. Core Explanation

Temporary tables are session-scoped structures that behave like normal tables but are automatically removed when:

- Your database connection ends (default in PostgreSQL/MySQL).
- Your transaction ends, if you create them inside a transaction block (depending on the database).

They let you:

- Load intermediate results for complex queries.
- Speed up multi-step ETL processes.
- Avoid naming collisions by relying on session-specific table names.

### 3. Syntax & Variants

### PostgreSQL & MySQL

```sql
-- Creates a temp table visible only in your session
CREATE TEMPORARY TABLE temp_sales (
  sale_id    INT,
  sale_date  DATE,
  amount     DECIMAL(10,2)
);

-- Optionally specify ON COMMIT behavior (PostgreSQL)
CREATE TEMPORARY TABLE temp_sales (
  sale_id   INT,
  sale_date DATE,
  amount    DECIMAL(10,2)
) ON COMMIT DROP;
```

### SQL Server

```sql
-- Local temp table: visible only in your session
CREATE TABLE #temp_sales (
  sale_id   INT,
  sale_date DATE,
  amount    DECIMAL(10,2)
);

-- Global temp table: visible to all sessions, auto-dropped when last session ends
CREATE TABLE ##temp_sales (
  sale_id   INT,
  sale_date DATE,
  amount    DECIMAL(10,2)
);
```

### Variants & Options

- **`IF NOT EXISTS`** (MySQL/PostgreSQL):
    
    ```sql
    CREATE TEMPORARY TABLE IF NOT EXISTS temp_sales (...);
    ```
    
- **ON COMMIT** (PostgreSQL only):
    - `PRESERVE ROWS` (default)
    - `DELETE ROWS` clears data but keeps table
    - `DROP` removes table on commit
- **Column Constraints** and **Indexes** apply just like normal tables.

### 4. Real-World Use Cases

- **Staging ETL Steps**Load raw CSV data into a temp table, clean or transform it, then insert into your production table.
- **Complex Joins & Aggregations**Break a giant query into chunks: stage filtered data in a temp table to index it, then join with other tables for faster performance.
- **Ad-Hoc Analysis**Analysts spin up temp tables to test hypotheses on subsets of data without risking the production schema.
- **Stored Procedures**Use temp tables within procedures to hold intermediate results, ensuring each execution is isolated.

### 5. Practice Problems

Assume you have `orders(order_id, customer_id, order_date, total)`:

1. Create a temporary table `recent_orders` holding orders from the last 30 days.
    
    *Hint:* Use `CREATE TEMPORARY TABLE` plus a `SELECT … WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'` clause.
    
2. Load `recent_orders`, add an index on `customer_id`, then join with `customers` to get `customer_name`.
    
    *Hint:* Use `CREATE INDEX` on the temp table and then a `SELECT … FROM recent_orders JOIN customers …`.
    
3. In SQL Server, demonstrate creating a global temp table `##top_products` that stores the top 5 products by sales amount.
    
    *Hint:* Use `SELECT TOP 5 … INTO ##top_products`.
    

### 6. Visual Walkthrough

Starting with the `orders` table:

| order_id | customer_id | order_date | total |
| --- | --- | --- | --- |
| 10 | 42 | 2025-07-01 | 150.00 |
| 11 | 35 | 2025-07-15 | 80.00 |
| 12 | 42 | 2025-07-28 | 230.00 |
| 13 | 29 | 2025-06-20 | 120.00 |
1. **CREATE & LOAD**
    
    ```sql
    CREATE TEMPORARY TABLE recent_orders AS
    SELECT * FROM orders
     WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';
    ```
    
    Temp table contents:
    
    | order_id | customer_id | order_date | total |
    | --- | --- | --- | --- |
    | 11 | 35 | 2025-07-15 | 80.00 |
    | 12 | 42 | 2025-07-28 | 230.00 |
2. **Index & Join**
    
    ```sql
    CREATE INDEX idx_customer ON recent_orders(customer_id);
    
    SELECT r.order_id, c.name, r.total
      FROM recent_orders r
      JOIN customers c
        ON r.customer_id = c.customer_id;
    ```
    
3. **END OF SESSION**
    
    When your connection closes, `recent_orders` disappears automatically.
    

---

## Adding Comments to SQL

### 1. Quick Refresher

Comments are annotations in your code that the database engine ignores. They help you and your team understand, maintain, and debug SQL without changing its behavior.

### 2. Core Explanation

Comments serve three main purposes:

- Documentation: Explain *why* a complex join, filter, or calculation exists.
- Readability: Break long queries into logical sections.
- Debugging: Temporarily disable parts of a query without deleting code.

Because comments don’t run, they’re safe for collaboration and version control, letting you trace changes and decisions over time.

### 3. Syntax & Variants

### Single-Line Comments

```sql
-- This is a single-line comment
SELECT *
  FROM orders
 WHERE amount > 100;  -- filter high-value orders
```

- `-` starts a comment that runs until the end of the line.
- Common across most SQL dialects (PostgreSQL, SQL Server, Oracle).

### Inline Comments (MySQL)

```sql
# Another single-line comment style in MySQL
SELECT * FROM users;
```

### Multi-Line (Block) Comments

```sql
/*
  This is a block comment.
  It can span multiple lines.
  Use it to explain complex logic or
  temporarily disable large code sections.
*/
SELECT customer_id,
       SUM(amount) AS total_sales
  FROM sales
 GROUP BY customer_id;
```

- `/* ... */` encloses one or more lines.
- Supported in PostgreSQL, MySQL, SQL Server, Oracle.

### Common Patterns & Best Practices

- Place a header comment at top of scripts with author, date, and purpose.
- Use section dividers to split long queries:
    
    ```sql
    -- ================================
    -- 1. Filter recent orders
    -- ================================
    ```
    
- Avoid nesting `/* */` inside block comments—most engines won’t allow it.

### 4. Real-World Use Cases

- **Query Reviews & Handoffs**Annotate nonobvious business rules (e.g., why you exclude certain regions).
- **ETL Scripts**Document each transformation step in a staging pipeline.
- **Versioned Migrations**Header comments in migration files track who created a table or index, when, and why.
- **Debugging Complex Joins**Comment out one join at a time to isolate performance issues or logic bugs.

### 5. Practice Problems

1. Given the query below, add comments to explain each join and the final filter:
    
    ```sql
    SELECT o.order_id,
           c.customer_name,
           p.product_name,
           o.quantity * p.unit_price AS revenue
      FROM orders o
      JOIN customers c ON o.customer_id = c.customer_id
      JOIN products p  ON o.product_id  = p.product_id
     WHERE o.order_date >= '2025-01-01';
    ```
    
    Hint: Use `--` before each logical block.
    
2. Temporarily disable the `WHERE` clause in this query without deleting it, then re-enable it:
    
    ```sql
    SELECT *
      FROM sessions
     WHERE is_active = TRUE;
    ```
    
    Hint: Wrap it in `/* ... */`.
    
3. Create a script header comment for a migration that adds a `last_login` column to `users`, including author and date. Then write the `ALTER TABLE` statement below it.

### 6. Visual Walkthrough

**Before comments:**

```sql
SELECT *
  FROM orders
 WHERE status = 'shipped';
```

**After adding comments:**

```sql
-- Retrieve only shipped orders for the logistics dashboard
-- Author: Vybhav | Date: 2025-07-29

SELECT *
  FROM orders
 WHERE status = 'shipped';  -- status can be: pending, shipped, cancelled
```

1. The top two lines explain the query’s purpose and ownership.
2. The inline comment on the `WHERE` clause clarifies possible `status` values.

Comments improve clarity without changing the result set or execution plan.

---

## Data Modeling & ER Diagrams

### 1. Quick Refresher

A data model is the blueprint for how information is structured and stored in a database.

- Conceptual model: high-level entities and relationships, no implementation details.
- Logical model: adds attributes, keys, and constraints—still platform-agnostic.
- Physical model: concrete tables, data types, indexes, partitions in a specific RDBMS.

Entity–Relationship (ER) diagrams visualize the conceptual/logical layers, showing entities (tables), attributes (columns), and relationships (lines with cardinality markers).

### 2. Core Explanation

At its core, data modeling bridges domain knowledge and technical design:

- Entity: a real-world object or concept, e.g., Customer, Order.
- Attribute: property of an entity, e.g., customer.name, order.date.
- Key Attribute: uniquely identifies an entity instance (Primary Key).
- Relationship: association between entities, e.g., Customer places Order.
- Cardinality: minimum/maximum instances allowed in a relationship—1:1, 1:N, N:M.
- Participation: whether involvement is mandatory (solid line) or optional (dashed).

Two popular ER notation styles:

| Notation | Key Characteristics |
| --- | --- |
| Chen | Diamonds for relationships; ovals for attributes. |
| Crow’s Foot | Straight lines with “crow’s feet” for the N side. |

Mapping ER diagrams to tables follows systematic rules:

- 1:1 → combine tables or use unique foreign keys.
- 1:N → foreign key in the “many” side.
- N:M → junction (associative) table with foreign keys to both ends.

### 3. Syntax & Variants

### Defining Tables for a 1:N Relationship

```sql
-- Customers (1) ←── Orders (N)
CREATE TABLE customers (
  customer_id SERIAL PRIMARY KEY,
  name        VARCHAR(100) NOT NULL
);

CREATE TABLE orders (
  order_id    SERIAL PRIMARY KEY,
  customer_id INT NOT NULL
    REFERENCES customers(customer_id),
  order_date  DATE NOT NULL
);
```

### Modeling an N:M Relationship with a Junction Table

```sql
-- Students ↔ Courses (N:M)
CREATE TABLE students (
  student_id   SERIAL PRIMARY KEY,
  student_name VARCHAR(100) NOT NULL
);

CREATE TABLE courses (
  course_id    SERIAL PRIMARY KEY,
  course_title VARCHAR(200) NOT NULL
);

CREATE TABLE enrollments (
  student_id   INT NOT NULL
    REFERENCES students(student_id),
  course_id    INT NOT NULL
    REFERENCES courses(course_id),
  enrollment_date DATE,
  PRIMARY KEY (student_id, course_id)
);
```

### Composite Keys vs. Surrogate Keys

- Composite Key:
    
    ```sql
    PRIMARY KEY (student_id, course_id)
    ```
    
- Surrogate Key alternative:
    
    ```sql
    enrollment_id SERIAL PRIMARY KEY
    ```
    

### 4. Real-World Use Cases

- OLTP Systems: ER diagrams drive design in transactional apps (e-commerce, banking).
- Data Warehouses: Star and snowflake schemas are specialized ER variants optimized for reporting.
- Microservices: Bounded contexts use ER modeling to ensure clear service boundaries and data ownership.

| Model Type | Strengths | Trade-offs |
| --- | --- | --- |
| Normalized (3NF) | Eliminates redundancy; enforces integrity | More joins in queries |
| Star Schema | Simplifies reporting; fast reads | Data duplication in dimensions |
| Snowflake | Further normalization of dimensions | More joins vs. star schema |

### 5. Practice Problems

1. **Library System**
    - Entities: Books, Authors, Borrowers, Loans.
    - Relationships:
        - Book ↔ Author (N:M)
        - Borrower → Loan (1:N)
    - Tasks:
        - Draw a crow’s foot ERD.
        - Write `CREATE TABLE` statements, including a junction table for authorships.
2. **E-Commerce Schema**
    - Entities: Product, Category, Order, OrderItem, Customer.
    - Relationships:
        - Product → Category (N:1)
        - Order → OrderItem (1:N)
        - Customer → Order (1:N)
    - Tasks:
        - Map to a star schema: identify fact and dimension tables.
        - Write DDL for fact_sales and dim_customer tables.
3. **Participation Constraints**
    - Model whether an Order must have at least one OrderItem (mandatory) and whether a Product may lack a Category (optional).
    - Translate to SQL constraints (`NOT NULL` on FKs) or application-level checks.

### 6. Visual Walkthrough

### Example ER Diagram (Crow’s Foot)

```
┌──────────┐          places           ┌──────────┐
│ Customer │1─────────────────────────∞│  Order   │
│──────────│                         │──────────│
│ customer_id PK                    │ order_id PK│
│ name                              │ order_date │
└──────────┘                         └──────────┘

                             contains
                               ∞
                               │
                           ┌────────────┐
                           │ OrderItem  │
                           │────────────│
                           │ order_id FK│───┐
                           │ product_id FK│ │
                           │ quantity     │ │
                           └──────────────┘ │
                                             │
      ┌──────────┐        categorized in     │  ∞
      │ Product  │1──────────────────────────┘
      │──────────│
      │ product_id PK
      │ name
      └──────────┘
```

### Mapping to Tables

| Table | Columns | Notes |
| --- | --- | --- |
| customers | customer_id PK, name | Entity |
| orders | order_id PK, customer_id FK, order_date | FK enforces 1:N |
| products | product_id PK, name | Independent entity |
| order_items | order_id FK, product_id FK, quantity | Junction for order ↔ product |

This end-to-end workflow—from domain analysis to ER diagram to SQL DDL—ensures data integrity, performance, and maintainability.

---