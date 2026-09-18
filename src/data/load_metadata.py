import duckdb

conn = duckdb.connect(database='catalog.db', read_only=False)
cursor = conn.cursor()

with open('sql/schema.sql', 'r') as file:
    cursor.execute(file.read())
cursor.execute("SHOW TABLES;")
print(cursor.fetchall())

cursor.execute(
    "SELECT count(*) FROM read_json_auto('data/raw/listings/metadata/*.json', ignore_errors=true);")
fetch_result = cursor.fetchall()
print(fetch_result)
cursor.execute(
    "SELECT node[1] FROM read_json_auto('data/raw/listings/metadata/*.json') LIMIT 5;")
print(cursor.fetchall())

cursor.execute("TRUNCATE TABLE categories;")

cursor.execute("""
INSERT INTO categories (category_id, category_name, parent_category_id)
WITH raw_paths AS (
    SELECT DISTINCT node[1].node_name AS full_path
    FROM read_json_auto('data/raw/listings/metadata/*.json')
    WHERE node[1].node_name IS NOT NULL
      AND node[1].node_name LIKE '/Categories/%'
),
split_paths AS (
    SELECT full_path, str_split(trim(full_path, '/'), '/') AS cat_array
    FROM raw_paths
),
unpacked AS (
    SELECT 
        cat_array,
        generate_subscripts(cat_array, 1) AS idx,
        cat_array[generate_subscripts(cat_array, 1)] AS cat_name
    FROM split_paths
),
hierarchical_categories AS (
    SELECT DISTINCT
        '/' || array_to_string(cat_array[1:idx], '/') AS category_id,
        cat_name AS category_name,
        CASE 
            WHEN idx > 1 THEN '/' || array_to_string(cat_array[1:idx-1], '/')
            ELSE NULL 
        END AS parent_category_id
    FROM unpacked
)
SELECT category_id, category_name, parent_category_id 
FROM hierarchical_categories;
""")
cursor.execute("SELECT COUNT(*) AS total_inserted_categories FROM categories;")
fetch_result = cursor.fetchall()
print(fetch_result)

cursor.execute("SELECT * FROM categories LIMIT 10;")
fetch_result = cursor.fetchall()
print(fetch_result)

cursor.execute(
    "SELECT * FROM read_json_auto('data/raw/listings/metadata/*.json') LIMIT 1")
fetch_result = cursor.fetchall()
print(fetch_result)

cursor.execute("TRUNCATE TABLE products;")

cursor.execute("""
INSERT INTO products (product_id, title, category_id)
WITH unique_products AS (
    SELECT 
        item_id AS product_id,
        list_filter(item_name, x -> x.language_tag LIKE 'en%') AS en_names,
        node[1].node_name AS full_path
    FROM read_json_auto('data/raw/listings/metadata/*.json')
    WHERE item_id IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
    PARTITION BY item_id 
    ORDER BY 
        CASE WHEN node[1].node_name LIKE '/Categories/%' THEN 0 ELSE 1 END,
        node[1].node_name,
        item_id
) = 1
)
SELECT 
    p.product_id,
    en_names[1].value AS title,
    c.category_id
FROM unique_products p
LEFT JOIN categories c ON c.category_id = p.full_path
WHERE len(en_names) > 0;
""")

cursor.execute("SELECT COUNT(*) FROM products;")
fetch_result = cursor.fetchall()
print(fetch_result)

cursor.execute("TRUNCATE TABLE product_images;")

cursor.execute("""
WITH all_images AS (
    SELECT 
        item_id AS product_id,
        main_image_id AS image_id,
        1 AS image_order
    FROM read_json_auto('data/raw/listings/metadata/*.json')
    WHERE main_image_id IS NOT NULL

    UNION ALL

    SELECT 
        item_id AS product_id,
        UNNEST(other_image_id) AS image_id,
        2 AS image_order
    FROM read_json_auto('data/raw/listings/metadata/*.json')
    WHERE other_image_id IS NOT NULL
)
SELECT 
    all_images.image_id,
    product_id,
    i.path,
    image_order
FROM all_images
JOIN read_csv_auto('data/raw/abo-images-small/metadata/*.csv.gz') AS i ON all_images.image_id = i.image_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY all_images.image_id ORDER BY all_images.image_id) = 1;
""")

cursor.execute("SELECT COUNT(*) FROM product_images;")
print("Total Images Inserted:", cursor.fetchall())

cursor.execute("""
SELECT COUNT(DISTINCT p.product_id) AS invalid_product_count 
FROM products p
LEFT JOIN product_images i ON p.product_id = i.product_id
WHERE i.image_id IS NULL OR p.title IS NULL OR TRIM(p.title) = '';
""")

result = cursor.fetchall()
print("Invalid products count (missing title or image):", result[0][0])

cursor.execute("""
SELECT LOWER(TRIM(title)) AS clean_title, COUNT(*) AS title_count
FROM products
WHERE title IS NOT NULL 
GROUP BY LOWER(TRIM(title))
HAVING COUNT(*) > 1
ORDER BY title_count DESC;
""")

duplicate_titles = cursor.fetchall()
print("Number of unique duplicate titles found:", len(duplicate_titles))
print("Top 3 duplicate titles:", duplicate_titles[:3])

cursor.execute("""
WITH RECURSIVE category_tree AS (
    SELECT 
        category_id,
        category_name,
        parent_category_id,
        category_id AS root_category_id,
        1 AS depth
    FROM categories

    UNION ALL

    SELECT 
        tree.category_id,
        tree.category_name,
        parent.parent_category_id,
        parent.category_id AS root_category_id,
        tree.depth + 1 AS depth
    FROM category_tree tree
    JOIN categories parent ON tree.parent_category_id = parent.category_id
)
SELECT MAX(depth) FROM category_tree;
""")

max_depth = cursor.fetchone()
print("Maximum Category Hierarchy Depth:", max_depth[0])

cursor.execute("""
SELECT COUNT(*) AS categories_with_parent
FROM categories
WHERE parent_category_id IS NOT NULL;
""")

result = cursor.fetchone()
print("Number of categories with a parent category:", result[0])

cursor.execute("""
WITH price_stats AS (
    SELECT 
        product_id,
        title,
        category_id,
        price,
        AVG(price) OVER (PARTITION BY category_id) AS avg_price,
        STDDEV(price) OVER (PARTITION BY category_id) AS stddev_price
    FROM products
    WHERE price IS NOT NULL AND price > 0
)
SELECT 
    product_id,
    title,
    category_id,
    price,
    avg_price,
    stddev_price,
    ABS(price - avg_price) / NULLIF(stddev_price, 0) AS z_score
FROM price_stats
WHERE ABS(price - avg_price) > 3 * stddev_price
ORDER BY z_score DESC;""")
result = cursor.fetchall()
print("Number of outlier products based on price:", len(result))
if result:
    max_z_score = max(row[-1] for row in result)
    print("Maximum Z-score among outliers:", max_z_score)

max_iterations = 8  

for i in range(max_iterations):
    cursor.execute("""
        WITH category_counts AS (
            SELECT p.product_id, p.category_id, c.parent_category_id,
                   COUNT(*) OVER (PARTITION BY p.category_id) AS prod_count
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.category_id
        )
        SELECT COUNT(*) FROM category_counts
        WHERE prod_count < 50 AND parent_category_id IS NOT NULL
    """)
    to_move = cursor.fetchone()[0]

    if to_move == 0:
        break

    cursor.execute("""
        UPDATE products
        SET category_id = updated_categories.new_category_id
        FROM (
            WITH category_counts AS (
                SELECT 
                    p.product_id,
                    p.category_id,
                    c.parent_category_id,
                    COUNT(*) OVER (PARTITION BY p.category_id) AS prod_count
                FROM products p
                LEFT JOIN categories c ON p.category_id = c.category_id
            )
            SELECT 
                product_id,
                CASE 
                    WHEN prod_count < 50 AND parent_category_id IS NOT NULL THEN parent_category_id
                    ELSE category_id
                END AS new_category_id
            FROM category_counts
        ) AS updated_categories
        WHERE products.product_id = updated_categories.product_id
          AND products.category_id != updated_categories.new_category_id;
    """)

cursor.execute("""
    UPDATE products 
    SET category_id = NULL 
    WHERE category_id = '/Categories';
""")
cursor.execute("SELECT COUNT(*) FROM products WHERE category_id IS NULL;")
print("Number of products with category_id set to NULL:", cursor.fetchone()[0])

cursor.execute("""
SELECT 
    c.category_id, 
    c.category_name, 
    COUNT(p.product_id) AS product_count
FROM categories c 
JOIN products p ON c.category_id = p.category_id
GROUP BY c.category_id, c.category_name
HAVING COUNT(p.product_id) < 50
ORDER BY product_count ASC;
""")
sparse_categories = cursor.fetchall()
print("Number of sparse categories (<50 products):", len(sparse_categories))

cursor.execute("ALTER TABLE products ADD COLUMN split VARCHAR;")
cursor.execute("""
UPDATE products
SET split = partitioned.split
FROM (
    WITH distinct_title_category AS (
        SELECT DISTINCT LOWER(TRIM(title)) AS title_key, category_id
        FROM products
    ),
    title_representative AS (
        SELECT title_key, category_id AS rep_category,
               ROW_NUMBER() OVER (PARTITION BY title_key ORDER BY category_id) AS rn
        FROM distinct_title_category
    ),
    title_groups AS (
        SELECT title_key, rep_category
        FROM title_representative
        WHERE rn = 1
    ),
    group_split AS (
        SELECT 
            title_key,
            (ROW_NUMBER() OVER (PARTITION BY rep_category ORDER BY RANDOM()) * 1.0)
            / COUNT(*) OVER (PARTITION BY rep_category) AS rel_pos
        FROM title_groups
    ),
    group_split_labeled AS (
        SELECT 
            title_key,
            CASE 
                WHEN rel_pos <= 0.80 THEN 'train'
                WHEN rel_pos <= 0.90 THEN 'val'
                ELSE 'test'
            END AS split
        FROM group_split
    )
    SELECT p.product_id, gsl.split
    FROM products p
    JOIN group_split_labeled gsl 
      ON LOWER(TRIM(p.title)) = gsl.title_key
) AS partitioned
WHERE products.product_id = partitioned.product_id;
""")
cursor.execute("""SELECT split, COUNT(*) AS count
FROM products
GROUP BY split;
""")
result = cursor.fetchall()
print("Data split counts:", result)


conn.connection.close() if hasattr(conn, 'connection') else conn.close()
