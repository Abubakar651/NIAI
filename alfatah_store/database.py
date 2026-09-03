"""
database.py
Direct CSV database operations: read, write, update, delete, and atomic checkout.
"""

from datetime import datetime
import pandas as pd

PRODUCTS_FILE = "data/products.csv"
ORDERS_FILE = "data/orders.csv"


# 1. READ FUNCTIONS
def get_products():
    """Reads and returns all products from products.csv as a DataFrame."""
    return pd.read_csv(PRODUCTS_FILE)


def get_orders():
    """Reads and returns all orders from orders.csv. If empty or missing, returns empty DataFrame."""
    try:
        return pd.read_csv(ORDERS_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["order_id", "product_name", "quantity", "total_price", "date", "channel"])


# 2. INVENTORY MANAGEMENT (CRUD)
def add_product(name, category, price, stock):
    """Adds a new product record to products.csv with an auto-incremented product_id."""
    df = get_products()
    new_id = int(df["product_id"].max() + 1) if not df.empty else 1
    new_row = {
        "product_id": new_id,
        "name": name.strip(),
        "category": category,
        "price": int(price),
        "stock": int(stock)
    }
    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    updated_df.to_csv(PRODUCTS_FILE, index=False)


def update_product(product_name, new_price, new_stock):
    """Updates price and stock for an existing product in products.csv."""
    df = get_products()
    df.loc[df["name"] == product_name, ["price", "stock"]] = [int(new_price), int(new_stock)]
    df.to_csv(PRODUCTS_FILE, index=False)


def delete_product(product_name):
    """Deletes a product by name from products.csv."""
    df = get_products()
    df = df[df["name"] != product_name]
    df.to_csv(PRODUCTS_FILE, index=False)


# 3. LOW STOCK CHECKER
def get_low_stock(threshold=10):
    """Returns all products whose current stock is less than the threshold."""
    df = get_products()
    return df[df["stock"] < threshold]


# 4. ATOMIC MULTI-ITEM CHECKOUT & STOCK VALIDATION
def atomic_checkout(cart, channel):
    """
    Atomically checks stock for all items in the cart before purchasing.
    If ANY item lacks stock:
        - Entire checkout aborts.
        - No stock is deducted, no order is saved.
    If ALL items pass check:
        - Deducts stock for all items.
        - Records each item in orders.csv.
    Returns: (success: bool, data_or_error_message)
    """
    if not cart:
        return False, "Cart is empty."

    df_products = get_products()

    # Step A: Validate stock for EVERY item in cart first
    for item in cart:
        match = df_products[df_products["name"] == item["name"]]
        if match.empty:
            return False, f"Product '{item['name']}' not found in inventory."

        available_stock = int(match.iloc[0]["stock"])
        if item["quantity"] > available_stock:
            return False, f"Not enough stock for '{item['name']}'! (Available: {available_stock}, In Cart: {item['quantity']})"

    # Step B: All items passed validation -> Deduct stock & prepare order entries
    df_orders = get_orders()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_order_id = len(df_orders) + 1
    new_orders = []

    for index, item in enumerate(cart):
        prod_index = df_products[df_products["name"] == item["name"]].index[0]
        # Deduct stock
        df_products.loc[prod_index, "stock"] -= item["quantity"]

        # Prepare order record
        new_orders.append({
            "order_id": start_order_id + index,
            "product_name": item["name"],
            "quantity": item["quantity"],
            "total_price": item["price"] * item["quantity"],
            "date": now_str,
            "channel": channel
        })

    # Step C: Write changes to both CSV files atomically
    df_products.to_csv(PRODUCTS_FILE, index=False)
    updated_orders = pd.concat([df_orders, pd.DataFrame(new_orders)], ignore_index=True)
    updated_orders.to_csv(ORDERS_FILE, index=False)

    return True, new_orders