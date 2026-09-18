DROP TABLE IF EXISTS product_images;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
CREATE TABLE if not EXISTS categories (
    category_id text PRIMARY KEY ,
    category_name VARCHAR(255) NOT NULL,
    parent_category_id TEXT 
);
CREATE TABLE products (
    product_id text PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    category_id text REFERENCES categories(category_id),
    brand_id text,
    price DECIMAL(10, 2),
    primary_image_path VARCHAR(255) 
);
create TABLE if not EXISTS product_images (
    image_id text PRIMARY KEY,
    product_id text ,
    image_path VARCHAR(255) NOT NULL,
    image_order INT NOT NULL
);

