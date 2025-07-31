# SQL_c1_m4

## Modifying and Analyzing Data with SQL:
Working with Text Strings -

### Why Text Functions Matter

Text columns often contain messy, inconsistent, or embedded information (emails, codes, addresses). 

SQL’s built-in string functions let you clean, transform, and extract insights without exporting data. You can standardize formats, parse substrings, search patterns, and even aggregate on fragments.

### Common String Functions

| Function | Purpose | Syntax Example |
| --- | --- | --- |
| LENGTH(text) | Number of characters | `LENGTH(name)` |
| SUBSTRING(text FROM x FOR y) | Extract a portion | `SUBSTRING(email FROM 1 FOR POSITION('@' IN email)-1)` |
| TRIM([LEADING | TRAILING] chars FROM text) | Remove unwanted padding |
| CONCAT(str1, str2, …) | Concatenate multiple strings | `CONCAT(first_name, ' ', last_name)` |
| LOWER(text), UPPER(text) | Change case | `LOWER(city)` |
| REPLACE(text, old, new) | Search and replace substrings | `REPLACE(phone, '-', '')` |
| POSITION(sub IN text) | Find substring start position | `POSITION('2025' IN order_code)` |
| LPAD(text, length, pad) | Left-pad text to fixed width | `LPAD(id::text, 6, '0')` |
| RPAD(text, length, pad) | Right-pad text | `RPAD(sku, 10, 'X')` |
| SPLIT_PART(text, delim, n) *(Postgres)* | Split by delimiter, pick segment | `SPLIT_PART(path, '/', 3)` |
| REGEXP_REPLACE(text, pattern, replace) *(Postgres)* | Advanced pattern replace | `REGEXP_REPLACE(desc, '\s+', ' ')` |

### Practical Cleaning & Extraction Examples

1. Remove leading/trailing spaces from product codes
    
    ```sql
    UPDATE products
       SET sku = TRIM(sku);
    ```
    
2. Extract username and domain from emails
    
    ```sql
    SELECT
      SUBSTRING(email FROM 1 FOR POSITION('@' IN email)-1) AS username,
      SUBSTRING(email FROM POSITION('@' IN email)+1)    AS domain
    FROM users;
    ```
    
3. Standardize city names to title case (Postgres example)
    
    ```sql
    SELECT INITCAP(LOWER(city)) AS clean_city
      FROM addresses;
    ```
    
4. Zero-pad invoice numbers to 8 digits
    
    ```sql
    SELECT LPAD(invoice_no::text, 8, '0') AS punch_no
      FROM invoices;
    ```
    
5. Replace multiple spaces with a single space in descriptions
    
    ```sql
    SELECT REGEXP_REPLACE(description, '\s+', ' ', 'g') AS clean_desc
      FROM products;
    ```
    

### Pattern Matching & Regular Expressions

- **LIKE** / **ILIKE**: `%` for any chars, `_` for single char
    
    ```sql
    SELECT * FROM customers WHERE name ILIKE 'A%n_';
    ```
    
- **SIMILAR TO**: SQL-standard regex-like syntax
    
    ```sql
    SELECT sku FROM inventory WHERE sku SIMILAR TO 'X[0-9]{3}%';
    ```
    
- **POSIX Regex** (Postgres): `~`, `~*` for case-insensitive match
    
    ```sql
    SELECT code FROM events WHERE code ~ '^[A-Z]{2}\d{4}$';
    ```
    

### Analyzing Text Data

1. Compute average and max length of product names
    
    ```sql
    SELECT
      AVG(LENGTH(name)) AS avg_len,
      MAX(LENGTH(name)) AS max_len
    FROM products;
    ```
    
2. Find top 5 most common email domains
    
    ```sql
    SELECT domain, COUNT(*) AS cnt
      FROM (
        SELECT SUBSTRING(email FROM POSITION('@' IN email)+1) AS domain
          FROM users
      ) t
    GROUP BY domain
    ORDER BY cnt DESC
    LIMIT 5;
    ```
    
3. Group users by first three letters of postal code
    
    ```sql
    SELECT SUBSTRING(postal_code FROM 1 FOR 3) AS prefix,
           COUNT(*) AS num_users
      FROM addresses
    GROUP BY prefix
    ORDER BY num_users DESC;
    ```
    

### Practice Exercises

1. Clean phone numbers by stripping non-numeric chars and ensure 10 digits.
2. Extract year-month (`YYYY-MM`) from a timestamp string and count records per month.
3. Use regex to validate and list only U.S. state abbreviations (e.g., `CA`, `NY`).
4. Split a file path into directory, filename, and extension using `SPLIT_PART` or regex.

---

## Working with Date and Time Strings

### Why Date/Time Functions Matter

Date and time columns capture when events occur, but raw strings can be inconsistent or hard to compare. SQL’s date/time functions let you convert, normalize, and compute intervals right inside your queries. This keeps data pipelines efficient and ensures analyses—like trend tracking or SLA monitoring—stay accurate.

### Common Date/Time Functions

| Function | Purpose | Syntax Example |
| --- | --- | --- |
| CAST(expr AS DATE/TIMESTAMP) | Convert string to date or timestamp | `CAST('2025-07-31' AS DATE)` |
| TO_DATE(string, format) | Parse date string with a format mask (Postgres, Oracle) | `TO_DATE('31/07/2025', 'DD/MM/YYYY')` |
| TO_TIMESTAMP(string, format) | Parse timestamp string with time components | `TO_TIMESTAMP('2025-07-31 23:14', 'YYYY-MM-DD HH24:MI')` |
| TO_CHAR(date, format) | Format date or timestamp to string | `TO_CHAR(order_ts, 'YYYY-MM-DD HH24:MI:SS')` |
| EXTRACT(part FROM date) | Pull out year, month, day, hour, etc. | `EXTRACT(YEAR FROM order_date)` |
| DATE_TRUNC(part, date) | Round down to the specified precision | `DATE_TRUNC('month', signup_ts)` |
| AGE(timestamp1, timestamp2) | Interval between two timestamps (Postgres) | `AGE(now(), last_login)` |
| INTERVAL | Define a span of time | `order_date + INTERVAL '7 days'` |
| CURRENT_DATE / CURRENT_TIME | Get today’s date or now’s time | `SELECT CURRENT_DATE, CURRENT_TIMESTAMP;` |

### Practical Conversion & Formatting Examples

1. Convert a string to a date, then format back to `MM/DD/YYYY`
    
    ```sql
    SELECT
      TO_CHAR(
        TO_DATE(order_date_str, 'YYYY-MM-DD'),
        'MM/DD/YYYY'
      ) AS formatted_date
    FROM raw_orders;
    ```
    
2. Parse a full timestamp string with milliseconds
    
    ```sql
    SELECT
      TO_TIMESTAMP(log_ts, 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')
        AS parsed_ts
    FROM api_logs;
    ```
    
3. Standardize mixed formats in one column (Postgres)
    
    ```sql
    UPDATE events
       SET event_date = COALESCE(
         TO_DATE(raw_date, 'MM-DD-YYYY'),
         TO_DATE(raw_date, 'YYYY/MM/DD')
       );
    ```
    

### Date/Time Arithmetic & Differences

- Add days, hours, or even months using `INTERVAL`
    
    ```sql
    SELECT order_date + INTERVAL '30 days' AS due_date
      FROM invoices;
    ```
    
- Compute the age or difference between two timestamps
    
    ```sql
    SELECT
      AGE(shipped_ts, ordered_ts) AS transit_time
    FROM shipments;
    ```
    
- Subtract dates to get integer days difference
    
    ```sql
    SELECT
      (payment_date - invoice_date) AS days_to_pay
    FROM billing;
    ```
    

### Aggregation & Analysis on Dates

1. Count orders by quarter
    
    ```sql
    SELECT
      EXTRACT(YEAR FROM order_date) AS yr,
      EXTRACT(QUARTER FROM order_date) AS qtr,
      COUNT(*) AS order_count
    FROM orders
    GROUP BY yr, qtr
    ORDER BY yr, qtr;
    ```
    
2. Find the busiest hour of the day
    
    ```sql
    SELECT
      EXTRACT(HOUR FROM login_ts) AS hr,
      COUNT(*) AS logins
    FROM user_logins
    GROUP BY hr
    ORDER BY logins DESC
    LIMIT 1;
    ```
    
3. Calculate rolling 7-day average of daily sales (Postgres)
    
    ```sql
    SELECT
      sale_date,
      AVG(daily_total)
        OVER (
          ORDER BY sale_date
          ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_avg
    FROM (
      SELECT sale_date, SUM(amount) AS daily_total
      FROM sales
      GROUP BY sale_date
    ) sub;
    ```
    

### Practice Exercises

1. Given a `timestamp_str` column in `logs`, parse it and extract the minute bucket (`YYYY-MM-DD HH24:MI`).
2. Compute how many days each customer remained inactive by subtracting `last_order_date` from `CURRENT_DATE`.
3. Group user signups by calendar week (`YYYY-WW`) and plot growth.
4. Update a table with US-format dates (`MM/DD/YYYY`) into ISO format (`YYYY-MM-DD`).

---

## Date and Time String Examples in SQL

### 1. Common String Formats

These are typical ways dates and times appear in raw data:

- `2025-07-31`
- `31/07/2025`
- `July 31, 2025`
- `07:45:30`
- `23:17:05.123`
- `2025-07-31 23:17:05`
- `2025-07-31T23:17:05Z`
- `2025-07-31 15:17:05-08`
- `31-JUL-25 11.45PM`

### 2. Parsing Examples

| String Example | Function & Format Mask | SQL Snippet | Result Type |
| --- | --- | --- | --- |
| `2025-07-31` | TO_DATE(string, `'YYYY-MM-DD'`) | `TO_DATE('2025-07-31','YYYY-MM-DD')` | DATE |
| `31/07/2025` | TO_DATE(string, `'DD/MM/YYYY'`) | `TO_DATE('31/07/2025','DD/MM/YYYY')` | DATE |
| `July 31, 2025` | TO_DATE(string, `'Month DD, YYYY'`) | `TO_DATE('July 31, 2025','Month DD, YYYY')` | DATE |
| `07:45:30` | CAST(string AS TIME) | `CAST('07:45:30' AS TIME)` | TIME |
| `23:17:05.123` | TO_TIMESTAMP(string, `'HH24:MI:SS.MS'`) | `TO_TIMESTAMP('23:17:05.123','HH24:MI:SS.MS')` | TIMESTAMP |
| `2025-07-31 23:17:05` | TO_TIMESTAMP(string, `'YYYY-MM-DD HH24:MI:SS'`) | `TO_TIMESTAMP('2025-07-31 23:17:05','YYYY-MM-DD HH24:MI:SS')` | TIMESTAMP |
| `2025-07-31T23:17:05Z` | TO_TIMESTAMP(string, `'YYYY-MM-DD"T"HH24:MI:SS"Z"'`) | `TO_TIMESTAMP('2025-07-31T23:17:05Z','YYYY-MM-DD"T"HH24:MI:SS"Z"')` | TIMESTAMP WITH TIME ZONE |
| `2025-07-31 15:17:05-08` | TO_TIMESTAMP(string, `'YYYY-MM-DD HH24:MI:SSOF'`) (Postgres) | `TO_TIMESTAMP('2025-07-31 15:17:05-08','YYYY-MM-DD HH24:MI:SSOF')` | TIMESTAMP WITH TIME ZONE |
| `31-JUL-25 11.45PM` | TO_TIMESTAMP(string, `'DD-MON-YY HH12.MIPM'`) (Oracle/Postgres) | `TO_TIMESTAMP('31-JUL-25 11.45PM','DD-MON-YY HH12.MIPM')` | TIMESTAMP |

### 3. Formatting Back to Strings

Convert dates/timestamps into human-readable formats with `TO_CHAR`:

- `TO_CHAR(order_date,'MM/DD/YYYY')` → `07/31/2025`
- `TO_CHAR(login_ts,'Dy, DD Mon YYYY HH12:MI AM')` → `Thu, 31 Jul 2025 11:17 PM`
- `TO_CHAR(sale_ts,'YYYY/MM/DD"T"HH24:MI:SS')` → `2025/07/31T23:17:05`

### 4. Practice Exercises

1. Parse a VARCHAR column `signup_str` in formats `DD-MM-YYYY` or `YYYY/MM/DD`, choosing the non-NULL conversion.
2. Extract the timezone offset from strings like `'2025-07-31 15:17:05-08'` and convert to UTC.
3. Given `event_ts` as text `'20250731 231705'`, parse it using `TO_DATE`/`TO_TIMESTAMP`.
4. Format `current_timestamp` into `YYYY-WW` (year-week) strings and count records per week.

---

## SQL CASE Statements: Conditional Logic in Queries

### Why Use CASE

SQL CASE expressions let you embed conditional logic directly into SELECTs, WHERE clauses, ORDER BYs, or aggregations. Rather than export data for post-processing, you can:

- Translate raw values into human-readable categories
- Handle NULLs or outliers with defaults
- Perform conditional aggregates for reporting

### CASE Syntax Variants

| Type | Structure | Use Case |
| --- | --- | --- |
| Simple CASE | ```sql |  |

CASE expr

WHEN val1 THEN res1

WHEN val2 THEN res2

ELSE res_default

END

```
| Searched CASE    | ```sql
CASE
  WHEN cond1 THEN res1
  WHEN cond2 THEN res2
  ELSE res_default
END
``` | Evaluate independent boolean conditions          |

---

## Example: Categorize Order Size

```sql
SELECT
  order_id,
  amount,
  CASE
    WHEN amount < 50   THEN 'small'
    WHEN amount < 200  THEN 'medium'
    WHEN amount < 500  THEN 'large'
    ELSE 'enterprise'
  END AS size_category
FROM orders;
```

This maps each order’s numeric `amount` into a textual bucket.

### Example: Handle NULL and Unexpected Values

```sql
SELECT
  user_id,
  status_code,
  CASE status_code
    WHEN 1 THEN 'active'
    WHEN 2 THEN 'pending'
    WHEN 3 THEN 'suspended'
    ELSE 'unknown'
  END AS status_label
FROM user_status;
```

Unknown or NULL codes fall into the `unknown` bucket via the `ELSE`.

### Conditional Aggregation

```sql
SELECT
  COUNT(*) FILTER (WHERE gender = 'F') AS female_count,  -- Postgres filter clause
  COUNT(*) FILTER (WHERE gender = 'M') AS male_count,
  COUNT(*) FILTER (WHERE gender IS NULL) AS unknown_gender
FROM users;
```

Or using CASE for broader compatibility:

```sql
SELECT
  SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS female_count,
  SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) AS male_count
FROM users;
```

### CASE in ORDER BY and WHERE

- ORDER BY priority: rank tasks by urgency
    
    ```sql
    SELECT task, priority
    FROM tasks
    ORDER BY
      CASE priority
        WHEN 'high'   THEN 1
        WHEN 'medium' THEN 2
        WHEN 'low'    THEN 3
        ELSE 4
      END;
    ```
    
- WHERE with CASE to filter by dynamic criteria
    
    ```sql
    SELECT *
    FROM sales
    WHERE
      CASE
        WHEN region = 'APAC' THEN amount > 1000
        ELSE amount > 500
      END;
    ```
    

### Practice Exercises

1. In a `transactions` table, classify `transaction_type` codes (`'D'`, `'W'`, `'T'`) into `'deposit'`, `'withdrawal'`, or `'transfer'`.
2. Compute a “risk” score: if `credit_score` < 600 then `'high'`, `<750` then `'medium'`, else `'low'`.
3. Using conditional aggregation, count how many orders fall into each of the five size buckets from the earlier example.
4. Sort products so that those with NULL `release_date` come last, then ascending by date.

---

## SQL Views: Abstraction and Reusability

### Why Use Views

Views encapsulate complex queries under a simple name.

They let you

- Simplify repeated joins and filters
- Present consistent, business-friendly schemas
- Restrict columns or rows for security

### Types of Views

| View Type | Definition | Data Freshness |
| --- | --- | --- |
| Standard View | Virtual table defined by a SELECT query | Always up-to-date |
| Materialized View | Physical snapshot stored on disk | Needs periodic refresh |

### Creating a Standard View

1. Write your base query.
2. Wrap it in a `CREATE VIEW` statement.
3. Reference the view like a regular table.

```sql
CREATE VIEW active_customers AS
SELECT
  customer_id,
  first_name,
  last_name,
  last_order_date
FROM customers
WHERE last_order_date >= CURRENT_DATE - INTERVAL '1 year';
```

Querying the view:

```sql
SELECT *
FROM active_customers
WHERE last_name ILIKE 'S%';
```

### Creating a Materialized View (Postgres Example)

```sql
CREATE MATERIALIZED VIEW monthly_sales_summary AS
SELECT
  DATE_TRUNC('month', sale_date) AS month,
  SUM(amount) AS total_sales,
  COUNT(*)   AS order_count
FROM sales
GROUP BY month;
```

Refresh on demand:

```sql
REFRESH MATERIALIZED VIEW monthly_sales_summary;
```

### Benefits of Views

- Encapsulate business logic in the database layer
- Enhance security by exposing only needed columns
- Simplify client queries and reduce boilerplate
- Improve maintainability when underlying tables change

### Considerations and Limitations

- Performance depends on the underlying query complexity
- Standard views don’t store data; every query recomputes results
- Materialized views require manual or scheduled refreshes
- Some views are non-updatable—you can’t `INSERT` or `UPDATE` through them

### Practice Exercises

1. Create a view that shows customers with no orders in the last six months.
2. Build a materialized view for daily website traffic and schedule a nightly refresh.
3. Test updating data through a simple view; observe which columns can be changed.
4. Use a view to mask sensitive columns (e.g., show only hashed emails).

---

## Data Governance and Profiling

### Why Data Governance Matters

Data governance defines policies, roles, and processes that ensure data is accurate, secure, and used responsibly. Strong governance:

- Improves trust in analytics and ML
- Meets regulatory and privacy requirements
- Clarifies ownership and stewardship

### Core Pillars of Data Governance

| Pillar | Description |
| --- | --- |
| Stewardship | Assigning data owners and stewards |
| Quality | Defining metrics, SLAs, and remediation workflows |
| Metadata Management | Cataloging schemas, data dictionaries, lineage |
| Security & Privacy | Access controls, encryption, masking, audit trails |
| Compliance | Adhering to GDPR, HIPAA, SOX, industry regulations |

### What Is Data Profiling?

Data profiling is the process of examining datasets to:

- Surface anomalies (missing, inconsistent, or outlier values)
- Understand distributions and relationships
- Inform quality rules and cleansing steps

Profiling delivers a “health report” so you can automate checks and monitor drift over time.

### Profiling Dimensions and Metrics

| Dimension | Typical Metrics | SQL Example |
| --- | --- | --- |
| Completeness | NULL count, % present | `COUNT(*) - COUNT(col)` |
| Accuracy | Valid value checks (e.g., regex for emails) | `SUM(CASE WHEN email !~ '^[^@]+@[^@]+\.[^@]+$' THEN 1 ELSE 0 END)` |
| Uniqueness | Distinct count, duplicate records | `COUNT(col), COUNT(DISTINCT col)` |
| Distribution | Min, max, avg, stddev for numeric; frequency for categories | `MIN(val), MAX(val), AVG(val), STDDEV(val)` |
| Consistency | Cross-table foreign key validity | `SUM(CASE WHEN fk_id NOT IN (SELECT pk FROM parent) THEN 1 END)` |
| Patterns | Regex or length checks | `COUNT(*) FILTER (WHERE LENGTH(code) <> expected_len)` |

### SQL Techniques for Quick Profiling

1. **Null and Distinct Counts**
    
    ```sql
    SELECT
      COUNT(*) AS total_rows,
      COUNT(col) AS non_null_count,
      COUNT(DISTINCT col) AS unique_count
    FROM table_name;
    ```
    
2. **Value Distribution**
    
    ```sql
    SELECT col, COUNT(*) AS freq
      FROM table_name
    GROUP BY col
    ORDER BY freq DESC
    LIMIT 10;
    ```
    
3. **Statistical Summaries**
    
    ```sql
    SELECT
      MIN(num_col), MAX(num_col), AVG(num_col), STDDEV(num_col)
    FROM table_name;
    ```
    
4. **Pattern Validation (Postgres Regex)**
    
    ```sql
    SELECT
      SUM(CASE WHEN col !~ '^[A-Z]{3}\d{4}$' THEN 1 ELSE 0 END) AS invalid_codes
    FROM table_name;
    ```
    

### Python Tools for Profiling

| Tool | Description | Code Snippet |
| --- | --- | --- |
| pandas | Basic profiles with `describe()` and `value_counts()` | `df.describe(include='all')` |
| pandas-profiling | Generates detailed HTML reports | `from pandas_profiling import ProfileReport`<br>`ProfileReport(df)` |
| Great Expectations | Declarative data tests and documentation | `import great_expectations as ge`<br>`df = ge.from_pandas(df)` |
| pandera | Schema and validation on DataFrame | `import pandera as pa`<br>`schema = pa.DataFrameSchema({...})` |

### Profiling Workflow

1. **Collect Metadata**
    - Catalog table names, column types, and source systems
2. **Run Automated Metrics**
    - Schedule SQL scripts or Python jobs to compute core metrics
3. **Analyze and Remediate**
    - Identify rule violations and route to data stewards
4. **Monitor Drift**
    - Compare current metrics to historical baselines
5. **Document and Report**
    - Store profiles in a metadata catalog or BI dashboard

### Best Practices

- Parameterize SQL profiling queries so they run on any table or column list.
- Integrate profiling jobs into your ETL or Airflow pipelines.
- Store profiles and anomalies in a central data catalog (e.g., Amundsen, DataHub).
- Set up automated alerts when metrics cross defined thresholds.
- Version your profiling scripts alongside code and schema changes.

### Practice Exercises

1. **Write a profiling SQL script** that takes any table name and outputs null counts, distinct counts, and top-5 values for each column.
2. **Generate a pandas profiling report** on a DataFrame loaded via `pandas.read_sql` and interpret key warnings.
3. **Define Great Expectations** rules for an `orders` table: order dates not in the future, amount ≥ 0, status in a fixed list.
4. **Implement drift monitoring**: store yesterday’s profiling metrics in a table, compare to today’s, and flag > 10% change.

---

## Using SQL for Data Science: Part 1

### Overview

In this first installment, we’ll build a solid foundation in SQL for data science workflows. You’ll learn to extract, clean, transform, and aggregate data—all within the database—so downstream analyses in Python or BI tools run faster and more reliably.

### 1. Environment Setup & Best Practices

- Choose your database
    - PostgreSQL, MySQL, SQL Server, or SQLite for prototyping
- Secure credentials
    - Use environment variables or secret managers
- Establish connections
    - Test with `psql`, MySQL CLI, or a GUI (e.g., DBeaver)
- Version control your scripts
    - Store `.sql` files alongside code

### 2. Basic Querying: SELECT, Filtering, Ordering

| Concept | Syntax Example | Notes |
| --- | --- | --- |
| Column selection | `SELECT col1, col2 FROM table;` | Avoid `SELECT *` in production |
| Row filtering | `WHERE condition` | Combine with `AND`, `OR`, `NOT` |
| Sorting | `ORDER BY col DESC, col2 ASC` | Default is ascending |
| Limiting results | `LIMIT 100` | Or `TOP 100` in SQL Server |

### 3. Data Cleaning & Transformation

- Handle NULLs
    - `COALESCE(col, default)`
    - `NULLIF(col, '')`
- String functions
    - `TRIM()`, `LOWER()`, `REPLACE()`
- Date/time functions
    - `TO_DATE()`, `DATE_TRUNC()`, `EXTRACT()`
- Conditional logic
    - `CASE` expressions to bucket or label

### 4. Aggregation & Grouping

```sql
SELECT
  category,
  COUNT(*)        AS record_count,
  AVG(amount)     AS avg_amount,
  MAX(created_at) AS last_seen
FROM sales
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY category
HAVING COUNT(*) > 10
ORDER BY avg_amount DESC;
```

Key points:

- `GROUP BY` defines aggregation buckets
- `HAVING` filters on aggregated values
- Combine aggregates with filtering for deeper insights

### 5. Joins & Subqueries

| Join Type | Description | Syntax Example |
| --- | --- | --- |
| INNER JOIN | Keep matching rows only | `FROM A INNER JOIN B ON A.id = B.a_id` |
| LEFT JOIN | All from A, match from B | `FROM A LEFT JOIN B ON …` |
| RIGHT JOIN | All from B, match from A | `FROM A RIGHT JOIN B ON …` |
| FULL OUTER | Union of LEFT and RIGHT | `FROM A FULL OUTER JOIN B ON …` |

Subqueries:

- Scalar: return single value in `SELECT` or `WHERE`
- Correlated: refer to outer query columns

### 6. Common Table Expressions (CTEs)

```sql
WITH recent_sales AS (
  SELECT * FROM sales
  WHERE sale_date >= CURRENT_DATE - INTERVAL '7 days'
),
ranked AS (
  SELECT
    product_id,
    RANK() OVER (ORDER BY amount DESC) AS sale_rank
  FROM recent_sales
)
SELECT * FROM ranked WHERE sale_rank <= 5;
```

Benefits:

- Break complex logic into readable steps
- Reuse intermediate results
- Support recursive CTEs for hierarchies

### Practice Exercises for Part 1

1. Connect to a sample database and list the top 20 customers by total spend.
2. Clean a `users` table: trim whitespace in `email`, fill `NULL` countries with `'Unknown'`.
3. Aggregate sales by weekday and find the busiest day.
4. Join `orders` with `order_items` to compute average items per order.
5. Use a CTE to isolate last month’s high-value transactions (amount > $1000) and rank them.

---

## Using SQL for Data Science: Part 2

### Overview

In this section, we dive into advanced SQL techniques that power data science workflows at scale. You’ll learn how to compute moving aggregates, percentiles, and statistical measures in-database. We’ll cover performance tuning—indexing, partitioning, materialized views—and show you how to integrate SQL-crafted features into Python for machine learning pipelines.

### 1. Window Functions & Advanced Aggregations

Window functions let you compute metrics over sliding or grouped partitions without collapsing rows. They’re essential for time-series features, ranking, and cohort analyses.

### Key Window Functions

| Function | Description |
| --- | --- |
| ROW_NUMBER() OVER (...) | Unique row number per partition |
| RANK() / DENSE_RANK() | Rank values with and without gaps |
| NTILE(n) | Divide partition into n buckets |
| LAG(col, offset) | Previous row’s value |
| LEAD(col, offset) | Next row’s value |
| FIRST_VALUE(col) | First value in window |
| LAST_VALUE(col) | Last value in window |
| SUM/AVG(...) OVER (...) | Cumulative or moving sums and averages |
| PERCENT_RANK() / CUME_DIST() | Relative standings |

### Example: Customer Recency, Frequency, Monetary (RFM)

```sql
WITH orders_ranked AS (
  SELECT
    customer_id,
    order_date,
    amount,
    ROW_NUMBER() OVER (
      PARTITION BY customer_id
      ORDER BY order_date DESC
    ) AS rn,
    MAX(order_date) OVER (
      PARTITION BY customer_id
    ) AS last_order_date,
    COUNT(*) OVER (
      PARTITION BY customer_id
    ) AS total_orders,
    SUM(amount) OVER (
      PARTITION BY customer_id
    ) AS total_spend
  FROM orders
)
SELECT
  customer_id,
  DATE_PART('day', NOW() - last_order_date) AS recency_days,
  total_orders AS frequency,
  total_spend AS monetary
FROM orders_ranked
WHERE rn = 1;
```

This produces one row per customer with recency, frequency, and monetary features.

### 2. Advanced Analytics & Statistical Functions

Beyond windowing, SQL engines offer built-in statistics:

- CORR(x, y) / COVAR_POP(x, y): correlation and population covariance
- STDDEV_POP(x) / VAR_POP(x): standard deviation and variance
- PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY col): continuous percentile
- PERCENTILE_DISC(p) WITHIN GROUP (ORDER BY col): discrete percentile
- APPROXIMATE_COUNT_DISTINCT(col) (e.g., HLL in BigQuery, HyperLogLog): fast distinct counts

### Example: Compute Median & 90th Percentile Order Value

```sql
SELECT
  PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY amount) AS median_order,
  PERCENTILE_CONT(0.9)  WITHIN GROUP (ORDER BY amount) AS p90_order
FROM orders
WHERE order_date >= DATE_TRUNC('month', CURRENT_DATE);
```

### 3. Performance Tuning & Query Optimization

Well-tuned queries are vital when datasets grow into hundreds of millions of rows.

### 3.1. Analyzing Query Plans

- Use `EXPLAIN ANALYZE <query>` to see actual runtime, row counts, and scan types.
- Identify sequential scans on large tables, expensive sorts, or nested loops.

### 3.2. Indexing Strategies

| Index Type | Use Case |
| --- | --- |
| B-Tree | Equality/range filters on scalar columns |
| Hash | Pure equality filters (Postgres-specific) |
| GIN / GiST | Full-text search, JSONB, array containment |
| Functional | Expressions (e.g., `LOWER(email)`) |
| Partial | Subset of rows (e.g., `WHERE active = TRUE`) |

**Tip**: Always test index impact with `EXPLAIN ANALYZE`. Too many indexes slow down writes.

### 3.3. Table Partitioning

- Range partitioning on dates splits hot data from cold.
- List or hash partitions for categorical or evenly spread keys.
- Query planner can prune partitions when predicates match.

### 4. Materialized Views & Pre-Aggregation

Materialized views store the results of expensive aggregations or joins, refreshed on-demand or schedule.

### Creating & Refreshing (Postgres)

```sql
CREATE MATERIALIZED VIEW daily_sales_summary AS
SELECT
  sale_date,
  COUNT(*) AS order_count,
  SUM(amount) AS total_sales
FROM sales
GROUP BY sale_date;

-- Manual refresh
REFRESH MATERIALIZED VIEW daily_sales_summary;
```

**Best Practice**: Schedule nightly or incremental refreshes via cron/Airflow. Use `CONCURRENTLY` in Postgres to avoid locking readers.

### 5. Integrating SQL with Python for Feature Engineering

Pull advanced SQL features directly into Python to feed ML models, reducing memory overhead and leveraging the database’s compute.

### 5.1. Using pandas.read_sql

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host/db")

sql = """
WITH rfm AS (
  -- recency/frequency/monetary CTE from earlier
) SELECT * FROM rfm;
"""

df_features = pd.read_sql(sql, con=engine, parse_dates=['last_order_date'])
```

### 5.2. Dynamic Query Composition with SQLAlchemy Core

```python
from sqlalchemy import Table, MetaData, select, func

metadata = MetaData(bind=engine)
orders = Table('orders', metadata, autoload_with=engine)

last_order = func.max(orders.c.order_date).label('last_order_date')
total_spend = func.sum(orders.c.amount).label('total_spend')

stmt = (
    select(
      orders.c.customer_id,
      last_order,
      total_spend
    )
    .group_by(orders.c.customer_id)
)

df = pd.read_sql(stmt, con=engine)
```

### 5.3. Feature Store Patterns

- Store feature queries as views or tables in the database.
- Use dbt to version, test, and document SQL transformations.
- Schedule feature extraction jobs, then materialize into feature store tables.

### 6. End-to-End Example: Predicting Churn

1. **Data Preparation (SQL)**
    - Compute RFM features, rolling engagement metrics, and tenure in CTEs and window functions.
2. **Pull into Python**
    - Use `pandas.read_sql` to load features and label data.
3. **Feature Engineering (Python)**
    - One-hot encode categorical columns, scale numeric features.
4. **Model Training**
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    X = df_features.drop(columns=['customer_id', 'churned'])
    y = df_features['churned']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    ```
    
5. **Inference in SQL or Python**
    - For batch scoring, write predictions back to the DB via SQLAlchemy.
    - For real-time, wrap SQL feature queries in an API endpoint.

---