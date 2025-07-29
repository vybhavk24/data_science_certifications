# SQL_c1_m2

## Basics of Filtering with SQL

### 1. Quick Refresher

Filtering means narrowing down rows in your result set. We use the `WHERE` clause in a `SELECT` statement to keep only the rows that meet your criteria.

No prior filtering knowledge needed—just remember that tables hold all rows, and `WHERE` picks a subset.

### 2. Core Explanation

The `WHERE` clause tests each row against one or more conditions. Only rows for which the condition is true get returned. This is powerful because:

- It pushes data reduction to the database engine, minimizing data transferred to your app.
- Complex logic—combining conditions with `AND`, `OR`, and `NOT`—lets you slice data precisely.
- Filtering happens before sorting, grouping, or aggregation, making queries efficient.

### 3. Syntax & Variants

### Basic Pattern

```sql
SELECT column_list
  FROM table_name
 WHERE condition;
```

- `SELECT column_list`: columns you want.
- `FROM table_name`: source table.
- `WHERE condition`: boolean expression to filter rows.

### Common Operators

- Comparison: `=`, `<>` (not equal), `>`, `<`, `>=`, `<=`
- Range: `BETWEEN value1 AND value2`
- Set membership: `IN (val1, val2, …)`
- Pattern matching: `LIKE 'prefix%'`, `ILIKE` (case-insensitive in PostgreSQL)
- Null check: `IS NULL`, `IS NOT NULL`
- Logical: `AND`, `OR`, `NOT`

### Examples & Breakdown

1. Equality & comparison
    
    ```sql
    SELECT *
      FROM employees
     WHERE salary > 80000;
    ```
    
    - `salary > 80000`: keeps rows where the salary column is greater than 80,000.
2. Range filtering
    
    ```sql
    SELECT id, order_date, total
      FROM orders
     WHERE order_date BETWEEN '2025-01-01' AND '2025-06-30';
    ```
    
    - `BETWEEN … AND …`: inclusive range filter on dates.
3. List membership
    
    ```sql
    SELECT name, department
      FROM employees
     WHERE department IN ('Engineering', 'Marketing');
    ```
    
    - `IN (…)`: shorthand for multiple equality checks.
4. Pattern matching
    
    ```sql
    SELECT email
      FROM users
     WHERE email LIKE '%@example.com';
    ```
    
    - `%`: wildcard in `LIKE`, matching any sequence of characters.
5. Combining conditions
    
    ```sql
    SELECT *
      FROM sales
     WHERE amount >= 100
       AND sale_date >= CURRENT_DATE - INTERVAL '30 days';
    ```
    
    - `AND`: both conditions must be true.

### 4. Real-World Use Cases

- **Data Analysis**Analysts filter high-value customers (`WHERE total_spent > 1000`) to target promotions.
- **Dashboards & Reports**Dashboards show last month’s sales via `WHERE sale_date >= …`, keeping live updates fast.
- **ML Feature Engineering**Data scientists filter logs to a relevant time window before aggregating features, reducing noise.
- **API Pagination & Security**Backend services include `WHERE user_id = ?` to enforce row-level access control.

Compared to client-side filtering (e.g., pandas), SQL filtering leverages indexes and query optimizers, running at database speed on large tables.

### 5. Practice Problems

Assume this `products` table:

| product_id | name | category | price | stock |
| --- | --- | --- | --- | --- |
| 1 | T-shirt | Apparel | 25.0 | 100 |
| 2 | Coffee Mug | Kitchen | 12.5 | 50 |
| 3 | Notebook | Stationery | 5.0 | 200 |
| 4 | Hoodie | Apparel | 45.0 | 30 |
| 5 | Pen | Stationery | 1.5 | 500 |
1. Retrieve all **Apparel** products priced between $20 and $50.
    
    *Hint:* Use `WHERE category = 'Apparel' AND price BETWEEN 20 AND 50`.
    
2. List products with names ending in “book” or “pen”, case-insensitive.
    
    *Hint:* Use `ILIKE '%book'` and combine with `OR`.
    
3. Find items that are either out of stock or low stock (fewer than 20).
    
    *Hint:* `WHERE stock = 0 OR stock < 20`.
    

### 6. Visual Walkthrough

Starting `products` table:

| product_id | name | category | price | stock |
| --- | --- | --- | --- | --- |
| 1 | T-shirt | Apparel | 25.0 | 100 |
| 2 | Coffee Mug | Kitchen | 12.5 | 50 |
| 3 | Notebook | Stationery | 5.0 | 200 |
| 4 | Hoodie | Apparel | 45.0 | 30 |
| 5 | Pen | Stationery | 1.5 | 500 |

Example query:

```sql
SELECT name, price, stock
  FROM products
 WHERE category = 'Apparel'
   AND price BETWEEN 20 AND 50;
```

1. **FROM** selects all rows.
2. **WHERE category = 'Apparel'** filters to rows 1 and 4.
3. **AND price BETWEEN 20 AND 50** keeps both, since 25.0 and 45.0 fall in the range.

Filtered result:

| name | price | stock |
| --- | --- | --- |
| T-shirt | 25.0 | 100 |
| Hoodie | 45.0 | 30 |

This is how SQL pushes filtering work to the database, returning only relevant rows for your analysis.

---

## Advanced Filtering with IN, OR, NOT

### 1. Why These Operators Matter

Filtering with `IN`, `OR`, and `NOT` gives you fine-grained control over which rows to include or exclude. These operators help you:

- Match against lists of values
- Combine multiple criteria flexibly
- Invert conditions without rewriting complex logic

### 2. Operator Syntax & Behavior

### IN

Checks if a column’s value is one of a specified list. Equivalent to multiple `= … OR = …` checks, but more concise.

```sql
column_name IN (value1, value2, value3)
```

### OR

Joins two or more boolean expressions. Returns rows where **any** condition is true.

```sql
condition1
  OR condition2
  OR condition3
```

### NOT

Negates a condition, filtering out rows where the expression is true.

```sql
NOT (condition)
```

Use with `IN` for exclusion:

```sql
column_name NOT IN (value1, value2)
```

### 3. Combining Conditions & Precedence

When you mix `AND`, `OR`, and `NOT`, SQL applies these rules in order:

1. `NOT` (highest priority)
2. `AND`
3. `OR` (lowest priority)

Use parentheses to override default precedence:

```sql
WHERE (A OR B)
  AND NOT C
```

### 4. Examples & Breakdown

Assume this `employees` table:

| emp_id | name | dept | salary | status |
| --- | --- | --- | --- | --- |
| 1 | Alice | Engineering | 120000 | active |
| 2 | Bob | Marketing | 80000 | active |
| 3 | Carol | Engineering | 95000 | terminated |
| 4 | Dave | Sales | 60000 | active |
| 5 | Eve | Marketing | 90000 | active |
1. Using `IN` to match multiple departments:
    
    ```sql
    SELECT name, dept
      FROM employees
     WHERE dept IN ('Engineering', 'Sales');
    ```
    
    Keeps rows where `dept` is either Engineering or Sales.
    
2. Achieving the same with `OR`:
    
    ```sql
    SELECT name, dept
      FROM employees
     WHERE dept = 'Engineering'
        OR dept = 'Sales';
    ```
    
3. Excluding terminated employees with `NOT`:
    
    ```sql
    SELECT *
      FROM employees
     WHERE NOT status = 'terminated';
    ```
    
    Alternatively:
    
    ```sql
    WHERE status <> 'terminated'
    ```
    
4. Excluding multiple statuses with `NOT IN`:
    
    ```sql
    SELECT name, status
      FROM employees
     WHERE status NOT IN ('terminated', 'on leave');
    ```
    
5. Mixing `OR` and `NOT` with parentheses:
    
    ```sql
    SELECT *
      FROM employees
     WHERE (dept = 'Marketing' OR dept = 'Sales')
       AND NOT status = 'terminated';
    ```
    
    First filters to Marketing/Sales, then removes terminated rows.
    

### 5. Real-World Scenarios

- Row-level security: restrict users to certain regions or roles
- Dynamic filters: supply a list of IDs to `IN` from your application
- Exclusion lists: block spam domains in email tables
- Cleanup scripts: delete records not matching a whitelist

### 6. Practice Problems

Using the above `employees` table:

1. Retrieve all **active** employees in either Engineering or Marketing.
2. List employees who are **not** in Sales or whose salary is **below** $90,000.
3. Find employees with `emp_id` in the set (2, 4, 5) **and** status not equal to `terminated`.
4. Exclude everyone from Engineering **and** Marketing in a single query.

---

## Using Wildcards in SQL

### 1. What Are Wildcards?

Wildcards let you match patterns within text columns. Instead of exact equality, you can search for substrings, single characters, or character ranges. This is key when you don’t know the full value or need flexible text filtering.

### 2. Core Syntax

Use the `LIKE` operator (and `ILIKE` in PostgreSQL for case-insensitive matching):

```sql
column_name LIKE 'pattern'
column_name ILIKE 'pattern'    -- Postgres only, ignores case
```

To handle literal `%` or `_`, add an `ESCAPE` clause:

```sql
column_name LIKE '%50\%%' ESCAPE '\'
```

### 3. Wildcard Symbols

| Symbol | Meaning | Example Pattern | Matches |
| --- | --- | --- | --- |
| % | Any sequence of zero or more characters | `'%car%'` | “car”, “scarce”, “racecar” |
| _ | Exactly one character | `'B_n_'` | “Bone”, “Band”, “Bono” |
| [ ] | Any single character within a range or list (SQL Server, Oracle) | `'[A-C]at'` | “Aat”, “B at”, “Cat” |
| [^ ] | Any single character *not* in the list (SQL Server) | `'[^0-9]%'` | Strings not starting with a digit |

### 4. Pattern Examples

1. Find emails at a certain domain (case-sensitive):
    
    ```sql
    SELECT email
      FROM users
     WHERE email LIKE '%@example.com';
    ```
    
2. Match any 5-character product codes starting with “X”:
    
    ```sql
    SELECT code
      FROM products
     WHERE code LIKE 'X____';
    ```
    
3. Exclude names beginning with a digit (SQL Server):
    
    ```sql
    SELECT name
      FROM contacts
     WHERE name LIKE '[^0-9]%';
    ```
    
4. Search for literal wildcard characters:
    
    ```sql
    SELECT note
      FROM logs
     WHERE note LIKE '%100\%%' ESCAPE '\';
    ```
    

### 5. Real-World Use Cases

- **Data Cleansing**: Isolate rows with unexpected characters or formats.
- **User Search**: Autocomplete “starts with” or “contains” features.
- **Audit & Compliance**: Find references to terms in free-text fields.
- **Logs Analysis**: Track error codes using consistent prefixes.

Compared to full-text search, `%…%` scans can be slow on large tables. Consider indexes on patterns like `col LIKE 'prefix%'` for better performance.

### 6. Practice Problems

Assume a `files` table:

| file_id | filename | description |
| --- | --- | --- |
| 1 | report_2025.pdf | Annual report |
| 2 | data_backup_01.zip | Monthly backup |
| 3 | summary_2024.docx | Yearly summary |
| 4 | notes_final.txt | Final meeting notes |
| 5 | README.md | Project readme |
1. Retrieve all `.txt` files.
2. List filenames containing exactly one underscore (`_`).
3. Find descriptions *not* ending in “report”.
4. Select filenames matching the pattern `'_backup_%'`.

### 7. Beyond LIKE: Regex & Performance

- PostgreSQL’s `~` and `~*` let you use full regular expressions for advanced patterns.
- MySQL’s `REGEXP` supports Perl-style regex.
- To speed up wildcard searches, add a full-text index or use trigram indexes (PostgreSQL’s `pg_trgm`).
- For very large text searches, dedicated search engines (Elasticsearch, Solr) often outperform SQL wildcards.

---

## Sorting with ORDER BY

### Why ORDER BY Matters

Sorting organizes query results in a defined sequence, making data easier to read, compare, and analyze. Databases can leverage indexes to sort efficiently, minimizing the need for post-processing in your application. Proper sorting is crucial for reports, pagination, rankings, and any scenario where order conveys meaning.

### Core Syntax

```sql
SELECT column_list
  FROM table_name
 ORDER BY column1 [ASC|DESC];
```

- `ORDER BY`: introduces the sort clause.
- `column1`: the column or expression to sort on.
- `ASC` (ascending, default) or `DESC` (descending) specifies direction.

### Multi-Column Sorting

You can sort by multiple columns to establish primary, secondary, tertiary order:

```sql
SELECT *
  FROM employees
 ORDER BY department ASC, salary DESC, name ASC;
```

1. Sorts by `department` alphabetically.
2. Within each department, orders by highest `salary` first.
3. Ties on salary break by `name` alphabetically.

### Handling NULLs and Expressions

- By default, many databases treat `NULL` as the lowest value, but some (Oracle) treat it highest.
- To override, use `NULLS FIRST` or `NULLS LAST`:

```sql
ORDER BY updated_at DESC NULLS LAST
```

- You can sort on expressions, e.g., functions or arithmetic:

```sql
ORDER BY COALESCE(completed_at, created_at) ASC
```

### Example & Visual Walkthrough

Assume a `products` table:

| id | name | category | price | rating |
| --- | --- | --- | --- | --- |
| 1 | T-shirt | Apparel | 25.0 | 4.1 |
| 2 | Coffee Mug | Kitchen | 12.5 | 4.7 |
| 3 | Notebook | Stationery | 5.0 | 3.9 |
| 4 | Hoodie | Apparel | 45.0 | 4.8 |
| 5 | Pen | Stationery | 1.5 | 4.2 |

Query:

```sql
SELECT name, category, price, rating
  FROM products
 ORDER BY category ASC, rating DESC;
```

Result:

| name | category | price | rating |
| --- | --- | --- | --- |
| Hoodie | Apparel | 45.0 | 4.8 |
| T-shirt | Apparel | 25.0 | 4.1 |
| Coffee Mug | Kitchen | 12.5 | 4.7 |
| Pen | Stationery | 1.5 | 4.2 |
| Notebook | Stationery | 5.0 | 3.9 |

### Real-World Use Cases

- Leaderboards: rank users by score and then by time.
- Pagination: maintain stable sort across pages (`ORDER BY id`).
- Reporting: list top‐selling products first, then by revenue.
- Dashboards: highlight most recent entries using timestamps.

### Practice Problems

1. Retrieve all employees sorted by hire date (newest first), then by last name.
2. List orders with priority “high” first, then by total amount descending.
3. Show products sorted by whether `stock = 0` (out of stock first), then price ascending.
4. Order a sales report by region, then by revenue growth (calculated field).

---

## Math Operations in SQL

### 1. Why Math Operations Matter

Mathematical expressions let you derive new insights—calculating profit, growth rates, ratios, or converting units—all within your queries. Pushing these computations to the database streamlines pipelines and keeps data transformations in one place.

### 2. Core Arithmetic Operators

- `+` : addition
- : subtraction
- : multiplication
- `/` : division
- `%` : modulo (remainder)

```sql
SELECT
  quantity * price      AS total_sales,
  price - cost           AS profit,
  quantity % 3           AS leftover_units
FROM order_items;
```

### 3. Built-in Math Functions

| Function | Description | Example |
| --- | --- | --- |
| `ABS(x)` | Absolute value | `ABS(-5.3)` → 5.3 |
| `CEIL(x)` / `CEILING(x)` | Smallest integer ≥ x | `CEIL(4.2)` → 5 |
| `FLOOR(x)` | Largest integer ≤ x | `FLOOR(4.8)` → 4 |
| `ROUND(x[, d])` | Round to d decimal places (default 0) | `ROUND(3.14159, 2)` → 3.14 |
| `POWER(x, y)` | x raised to the y-th power | `POWER(2, 3)` → 8 |
| `SQRT(x)` | Square root | `SQRT(9)` → 3 |
| `LOG(x)` / `LN(x)` | Natural logarithm (base e) | `LOG(10)` → ~2.3026 |
| `LOG10(x)` | Base-10 logarithm | `LOG10(100)` → 2 |
| `EXP(x)` | e raised to the x-th power | `EXP(1)` → ~2.7183 |
| `TRUNC(x, d)` | Truncate to d decimal places | `TRUNC(3.987, 1)` → 3.9 |

### 4. Numeric Promotion & Casting

SQL often promotes integers to floats when mixing types, but you can cast explicitly:

```sql
SELECT
  5   / 2      AS int_division,    -- → 2  (integer division in some dialects)
  5.0 / 2      AS float_division,  -- → 2.5
  CAST(5 AS DECIMAL(5,2)) / 2     AS precise_division;
```

Use `CAST(col AS NUMERIC(p, s))` or dialect-specific syntax (`::NUMERIC`) to control precision.

### 5. Example & Visual Walkthrough

Assume a `products` table:

| product_id | name | price | cost |
| --- | --- | --- | --- |
| 1 | T-shirt | 25.0 | 10.0 |
| 2 | Hoodie | 45.0 | 20.0 |
| 3 | Mug | 12.5 | 5.0 |

Query to compute profit and margin:

```sql
SELECT
  name,
  price,
  cost,
  price - cost AS profit,
  ROUND((price - cost) / price * 100, 1) AS margin_pct
FROM products;
```

Step-by-step:

1. Compute `price - cost` for profit.
2. Divide by `price`, multiply by 100 for percentage.
3. Round to one decimal place.

Result:

| name | price | cost | profit | margin_pct |
| --- | --- | --- | --- | --- |
| T-shirt | 25.0 | 10.0 | 15.0 | 60.0 |
| Hoodie | 45.0 | 20.0 | 25.0 | 55.6 |
| Mug | 12.5 | 5.0 | 7.5 | 60.0 |

### 6. Real-World Use Cases

- **Feature Engineering**: derive ratios (click-through rates, conversion rates) before modeling.
- **Financial Reports**: calculate year-over-year growth, ROI, compound interest.
- **Data Quality**: flag anomalies via `CASE WHEN ABS(actual - expected) > threshold THEN…`.
- **Unit Conversion**: e.g., seconds to hours (`dur_sec/3600`).

### 7. Practice Problems

Assume a `sales` table:

| id | revenue | cost | duration_sec |
| --- | --- | --- | --- |
| 1 | 1000.00 | 700.0 | 3600 |
| 2 | 500.00 | 450.0 | 1800 |
| 3 | 1200.00 | 600.0 | 5400 |
1. Compute `profit` and `profit_margin` (percentage, two decimals).
2. Convert `duration_sec` to hours (one decimal).
3. Flag rows where margin < 20% using `CASE`.
4. Calculate the geometric mean of profits: `POWER(product_of_profits, 1/count)`.

---

## Aggregate Functions in SQL

### 1. Why Aggregates Matter

Databases often hold millions of rows—aggregate functions let you collapse that detail into meaningful summaries.

You can compute totals, averages, counts, and extremes directly in SQL, pushing heavy lifting to the engine.

This keeps your application code lean and ensures analytics run at optimal database speed.

### 2. Core Aggregate Functions

| Function | Description | Example |
| --- | --- | --- |
| `COUNT(col)` | Number of non-null values | `COUNT(user_id)` |
| `COUNT(*)` | Total number of rows | `COUNT(*)` |
| `SUM(col)` | Sum of numeric column | `SUM(sales_amount)` |
| `AVG(col)` | Average of numeric column | `AVG(rating)` |
| `MIN(col)` | Smallest value | `MIN(order_date)` |
| `MAX(col)` | Largest value | `MAX(price)` |

### 3. GROUP BY & HAVING

`GROUP BY` clusters rows that share one or more column values.

Aggregate functions then collapse each group to a single row.

Use `HAVING` to filter on aggregates, since `WHERE` cannot reference them.

```sql
SELECT
  department,
  COUNT(*)        AS employee_count,
  AVG(salary)     AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 70000;
```

### 4. Example & Visual Walkthrough

Assume this `sales` table:

| sale_id | region | amount |
| --- | --- | --- |
| 1 | East | 1000 |
| 2 | West | 750 |
| 3 | East | 1250 |
| 4 | North | 500 |
| 5 | West | 1100 |

Query:

```sql
SELECT
  region,
  SUM(amount) AS total_sales,
  COUNT(*)    AS num_sales
FROM sales
GROUP BY region;
```

1. **GROUP BY region** splits into East, West, North.
2. **SUM(amount)** and *COUNT()*collapse each region.

Result:

| region | total_sales | num_sales |
| --- | --- | --- |
| East | 2250 | 2 |
| West | 1850 | 2 |
| North | 500 | 1 |

### 5. Real-World Use Cases

- Business intelligence: monthly revenue per product line.
- Monitoring: error counts by service and hour.
- ML feature prep: average session length per user.
- Compliance: minimum and maximum transaction values by branch.

### 6. Practice Problems

Using this `orders` table:

| order_id | customer_id | total_amount | status |
| --- | --- | --- | --- |
| 1 | 101 | 250.00 | complete |
| 2 | 102 | 75.00 | pending |
| 3 | 101 | 100.00 | complete |
| 4 | 103 | 500.00 | complete |
| 5 | 102 | 120.00 | pending |
1. Compute each customer’s total spend and order count.
2. Find average, minimum, and maximum `total_amount` for `status = 'complete'`.
3. List customers having spent over $300.
4. Show status-wise order counts but exclude statuses with fewer than 2 orders.

---

## Grouping Data with SQL

### Why Grouping Matters

Grouping lets you collapse detailed rows into summarized buckets by one or more keys.

It’s essential for reports and analytics—calculating totals, averages, counts, and other metrics per category without pulling all raw data into your application.

By pushing these summaries into the database, you leverage optimized engines and keep your codebase simpler.

### Core Syntax: GROUP BY and HAVING

Use `GROUP BY` to define one or more columns that partition your data into groups.

Aggregate functions like `SUM()`, `COUNT()`, `AVG()`, `MIN()`, and `MAX()` then operate on each group.

To filter groups by their aggregate values, use `HAVING`—`WHERE` cannot reference aggregates.

```sql
SELECT
  column1,
  AGG_FUNC(column2) AS aggregate_result
FROM table_name
GROUP BY column1
HAVING AGG_FUNC(column2) > threshold;
```

### Example & Visual Walkthrough

Assume a `transactions` table:

| txn_id | customer_id | amount | region |
| --- | --- | --- | --- |
| 1 | 101 | 120.0 | East |
| 2 | 102 | 75.0 | West |
| 3 | 101 | 100.0 | East |
| 4 | 103 | 200.0 | North |
| 5 | 102 | 125.0 | West |

Query: total spend and count of transactions per customer, but only for spenders above $150.

```sql
SELECT
  customer_id,
  COUNT(*)      AS num_txns,
  SUM(amount)   AS total_spent
FROM transactions
GROUP BY customer_id
HAVING SUM(amount) > 150;
```

Result:

| customer_id | num_txns | total_spent |
| --- | --- | --- |
| 101 | 2 | 220.0 |
| 102 | 2 | 200.0 |

### Real-World Use Cases

- Business reporting: monthly revenue per product line.
- Customer analytics: average order value by segment.
- Operations: count of error events by service and hour.
- ML feature engineering: number of sessions per user before modeling.

### Practice Problems

Given this `orders` table:

| order_id | user_id | status | total |
| --- | --- | --- | --- |
| 1 | 201 | complete | 150.00 |
| 2 | 202 | pending | 75.00 |
| 3 | 201 | complete | 100.00 |
| 4 | 203 | canceled | 200.00 |
| 5 | 202 | complete | 120.00 |
1. Compute each user’s total order count and sum of `total`.
2. Find status-wise average order total.
3. List users with more than one “complete” order.
4. Show statuses where the sum of `total` exceeds $180.

---

## SQL Across Data Science Languages

### 1. Why Integrate SQL?

Integrating SQL directly in your data science workflow lets you push heavy data filtering, aggregation, and joins to the database engine.

This keeps your in-memory environment lean, speeds up prototyping, and ensures consistency between exploratory analysis and production pipelines.

### 2. Python: pandas & SQLAlchemy

- Use **SQLAlchemy** to define your database engine:
    
    ```python
    from sqlalchemy import create_engine
    engine = create_engine("postgresql://user:pass@host:5432/dbname")
    ```
    
- Run ad-hoc queries with **pandas**:
    
    ```python
    import pandas as pd
    df = pd.read_sql("SELECT * FROM sales WHERE amount > 100", engine)
    ```
    
- Build parameterized queries safely:
    
    ```python
    query = "SELECT * FROM users WHERE signup_date >= :start"
    df = pd.read_sql(query, engine, params={"start": "2025-01-01"})
    ```
    

### 3. R: DBI, dplyr & sqldf

- **DBI** + **RPostgres** for robust connections:
    
    ```r
    library(DBI)
    con <- dbConnect(RPostgres::Postgres(), dbname = "dbname",
                     host = "host", user = "user", password = "pass")
    df <- dbGetQuery(con, "SELECT * FROM orders WHERE status='complete'")
    ```
    
- **dplyr**’s `dbplyr` translates verbs to SQL:
    
    ```r
    library(dplyr)
    tbl(con, "orders") %>%
      filter(status == "complete", total > 100) %>%
      summarise(avg = mean(total)) %>%
      collect()
    ```
    
- **sqldf** for in-memory SQLite queries on data frames:
    
    ```r
    library(sqldf)
    results <- sqldf("SELECT * FROM df WHERE col LIKE 'A%'")
    ```
    

### 4. Scala & Spark: Spark SQL

- Register DataFrame as a temp view:
    
    ```scala
    val df = spark.read.json("data.json")
    df.createOrReplaceTempView("logs")
    val result = spark.sql("SELECT user_id, count(*) FROM logs GROUP BY user_id")
    ```
    
- Mix DataFrame API and SQL seamlessly:
    
    ```scala
    import spark.implicits._
    val agg = df.filter($"amount" > 100)
                .groupBy("region").agg(avg("amount"))
    ```
    

### 5. Julia: SQLite.jl & Query.jl

- **SQLite.jl** for embedded databases:
    
    ```julia
    using SQLite, DataFrames
    db = SQLite.DB("data.db")
    df = DBInterface.execute(db, "SELECT * FROM table WHERE x > 5") |> DataFrame
    ```
    
- **Query.jl** for LINQ-style queries on DataFrames:
    
    ```julia
    using Query, DataFrames
    df |> @filter(_.col != missing) |> @map({x = _.col}) |> DataFrame
    ```
    

### 6. Connecting to Other Engines

Common drivers and URIs:

- MySQL: `mysql+pymysql://…` (Python), `RMySQL`
- SQL Server: `mssql+pyodbc://…` (Python), `odbc` in R
- Oracle: `cx_Oracle` (Python), `ROracle` in R

Always secure credentials—use environment variables or secret managers.

### 7. Best Practices

- Push filters, aggregations, and joins into SQL, not application code.
- Parameterize queries to avoid SQL injection.
- Use connection pooling for repeated queries.
- Leverage ORM only when object mapping saves you boilerplate; else prefer raw SQL for analytics.

---