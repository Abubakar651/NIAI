"""
app.py
Streamlit User Interface for Al-Fatah Super Store Management System.
Connects with database.py for all data operations.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

import database as db

# Streamlit Page Setup
st.set_page_config(page_title="Al-Fatah Super Store", page_icon="🛒", layout="wide")


# --- HELPER COMPONENTS ---

def show_low_stock_banner(threshold=10):
    """Displays a warning banner if any item stock is below threshold."""
    low_stock = db.get_low_stock(threshold)
    if not low_stock.empty:
        items_str = ", ".join([f"**{r['name']}** ({r['stock']} left)" for _, r in low_stock.iterrows()])
        st.warning(f"⚠️ **Low Stock Alert (< {threshold} units):** {items_str}")


def render_cart(cart_key, channel_label):
    """Renders the current multi-item cart, shows totals, and triggers checkout."""
    cart = st.session_state.get(cart_key, [])
    st.subheader(f"🛒 Current Cart ({len(cart)} items)")

    if not cart:
        st.info("Cart is currently empty. Add items above.")
        return None

    cart_df = pd.DataFrame(cart)
    cart_df["Subtotal (PKR)"] = cart_df["price"] * cart_df["quantity"]
    st.dataframe(cart_df[["name", "price", "quantity", "Subtotal (PKR)"]], use_container_width=True, hide_index=True)

    total_bill = cart_df["Subtotal (PKR)"].sum()
    st.markdown(f"### **Grand Total: PKR {total_bill:,}**")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button(f"Complete {channel_label} Checkout", type="primary", use_container_width=True):
            success, result = db.atomic_checkout(cart, channel=channel_label.lower())
            if success:
                st.session_state[cart_key] = []  # Clear cart on success
                return result
            else:
                st.error(result)

    with col2:
        if st.button("Clear Cart", use_container_width=True):
            st.session_state[cart_key] = []
            st.rerun()

    return None


# --- APP PAGES ---

# PAGE 1: HOME
def page_home():
    """Welcome page with quick role redirection."""
    st.title("🏪 Welcome to Al-Fatah Super Store")
    role = st.session_state["role"]
    st.write(f"Logged in as: **{role}**")

    if role == "Customer":
        st.info("Browse our fresh groceries and enjoy delivery straight to your doorstep!")
        if st.button("🛍️ Shop Online Now", type="primary"):
            st.session_state["nav_selection"] = "Shop Online"
            st.rerun()
    elif role == "Cashier":
        st.info("Point-of-Sale (POS) counter ready to process customer items and print receipts.")
        if st.button("🧾 Open Physical Billing", type="primary"):
            st.session_state["nav_selection"] = "Physical Billing"
            st.rerun()
    else:
        st.info("Admin Portal: Full control over inventory, low stock warnings, and sales analytics.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📦 Manage Products", use_container_width=True):
                st.session_state["nav_selection"] = "Products"
                st.rerun()
        with col2:
            if st.button("📋 View Orders & Analytics", use_container_width=True):
                st.session_state["nav_selection"] = "Orders"
                st.rerun()


# PAGE 2: PRODUCTS (Admin CRUD & Cashier Read-Only)
def page_products():
    """Product catalog view. Full CRUD for Admin, view-only for Cashier."""
    st.title("📦 Product Catalog")
    show_low_stock_banner()

    df = db.get_products()
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.session_state["role"] != "Admin":
        st.info("ℹ️ Read-Only View: Switch role to Admin in sidebar to add, edit, or delete products.")
        return

    tab_add, tab_edit, tab_del = st.tabs(["➕ Add Product", "✏️ Edit Price/Stock", "🗑️ Delete Product"])

    with tab_add:
        with st.form("form_add", clear_on_submit=True):
            name = st.text_input("Product Name")
            category = st.selectbox("Category", ["Grocery", "Dairy", "Bakery", "Snacks", "Beverages", "Detergent", "Personal Care", "Frozen Foods"])
            price = st.number_input("Price (PKR)", min_value=1, step=10, value=100)
            stock = st.number_input("Stock", min_value=1, step=5, value=20)
            if st.form_submit_button("Add Product"):
                if name.strip():
                    db.add_product(name, category, price, stock)
                    st.success(f"Product '{name}' added successfully!")
                    st.rerun()
                else:
                    st.warning("Product name cannot be empty.")

    with tab_edit:
        if not df.empty:
            chosen = st.selectbox("Select Product to Edit", df["name"].tolist(), key="edit_item")
            row = df[df["name"] == chosen].iloc[0]
            with st.form("form_edit"):
                new_price = st.number_input("Update Price", value=int(row["price"]), min_value=1, step=10)
                new_stock = st.number_input("Update Stock", value=int(row["stock"]), min_value=0, step=1)
                if st.form_submit_button("Save Changes"):
                    db.update_product(chosen, new_price, new_stock)
                    st.success(f"Updated '{chosen}' successfully!")
                    st.rerun()

    with tab_del:
        if not df.empty:
            chosen_del = st.selectbox("Select Product to Delete", df["name"].tolist(), key="del_item")
            if st.button("Confirm Delete", type="primary"):
                db.delete_product(chosen_del)
                st.success(f"Deleted '{chosen_del}' successfully!")
                st.rerun()


# PAGE 3: SHOP ONLINE (Customer Multi-Item Cart)
def page_shop_online():
    """Online shopping portal for customers with multi-item cart."""
    st.title("🛒 Shop Online")

    if "online_cart" not in st.session_state:
        st.session_state["online_cart"] = []

    df = db.get_products()
    st.dataframe(df[["name", "category", "price", "stock"]], use_container_width=True, hide_index=True)

    st.subheader("Add Items to Cart")
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        chosen_prod = st.selectbox("Select Product", df["name"].tolist(), key="online_item")
    with col2:
        prod_row = df[df["name"] == chosen_prod].iloc[0]
        qty = st.number_input(f"Quantity (Available: {prod_row['stock']})", min_value=1, value=1, step=1, key="online_q")
    with col3:
        st.write("")
        st.write("")
        if st.button("➕ Add to Cart", key="btn_add_online"):
            existing = next((i for i in st.session_state["online_cart"] if i["name"] == chosen_prod), None)
            if existing:
                existing["quantity"] += qty
            else:
                st.session_state["online_cart"].append({
                    "name": chosen_prod,
                    "price": int(prod_row["price"]),
                    "quantity": qty
                })
            st.success(f"Added {qty}x '{chosen_prod}' to your cart.")
            st.rerun()

    orders = render_cart("online_cart", channel_label="Online")
    if orders:
        total = sum(i["total_price"] for i in orders)
        st.success(f"🎉 Order placed successfully! {len(orders)} products purchased for PKR {total:,}.")
        st.balloons()


# PAGE 4: PHYSICAL BILLING (Cashier Counter POS)
def page_physical_billing():
    """POS billing counter for cashiers with instant printed receipt."""
    st.title("🧾 Physical Store Billing (POS)")
    show_low_stock_banner()

    if "pos_cart" not in st.session_state:
        st.session_state["pos_cart"] = []

    df = db.get_products()
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        chosen_prod = st.selectbox("Scan / Select Item", df["name"].tolist(), key="pos_item")
    with col2:
        prod_row = df[df["name"] == chosen_prod].iloc[0]
        qty = st.number_input(f"Qty (Available: {prod_row['stock']})", min_value=1, value=1, step=1, key="pos_q")
    with col3:
        st.write("")
        st.write("")
        if st.button("➕ Add to Bill", key="btn_add_pos"):
            existing = next((i for i in st.session_state["pos_cart"] if i["name"] == chosen_prod), None)
            if existing:
                existing["quantity"] += qty
            else:
                st.session_state["pos_cart"].append({
                    "name": chosen_prod,
                    "price": int(prod_row["price"]),
                    "quantity": qty
                })
            st.success(f"Added {qty}x '{chosen_prod}' to current bill.")
            st.rerun()

    orders = render_cart("pos_cart", channel_label="Physical")
    if orders:
        st.success("✅ Transaction Recorded Successfully!")
        receipt_items = "\n".join([f"{o['product_name'][:18]:<18} x{o['quantity']:<2} PKR {o['total_price']:,}" for o in orders])
        grand_total = sum(o["total_price"] for o in orders)

        st.markdown(
            f"""
            ```text
            ========================================
                     AL-FATAH SUPER STORE
                   OFFICIAL COUNTER RECEIPT
            ========================================
            Date: {datetime.now().strftime("%d-%b-%Y %I:%M %p")}
            Channel: In-Store POS (Counter)
            ----------------------------------------
            ITEM               QTY  TOTAL
            ----------------------------------------
{receipt_items}
            ----------------------------------------
            GRAND TOTAL:       PKR {grand_total:,}
            ========================================
                Thank you for visiting Al-Fatah!
            ```
            """
        )


# PAGE 5: ORDERS & ANALYTICS
def page_orders():
    """Order transaction table and sales analytics with channel filtering."""
    st.title("📋 Order History & Analytics")
    df = db.get_orders()

    if df.empty:
        st.info("No orders recorded yet.")
        return

    filter_option = st.radio("Filter Orders By Channel:", ["All", "online", "physical"], horizontal=True)
    filtered = df if filter_option == "All" else df[df["channel"] == filter_option]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions", len(filtered))
    col2.metric("Total Revenue", f"PKR {filtered['total_price'].sum():,}")
    col3.metric("Units Sold", f"{filtered['quantity'].sum():,}")


# --- MAIN ROUTER & ROLE ACCESS ---

def main():
    """App entry point handling role permissions and sidebar navigation."""
    st.sidebar.title("🔐 Role Access")
    role = st.sidebar.selectbox("Current Role:", ["Admin", "Cashier", "Customer"], key="role_picker")
    st.session_state["role"] = role

    # Page access rules by role
    role_permissions = {
        "Admin": ["Home", "Products", "Shop Online", "Physical Billing", "Orders"],
        "Cashier": ["Home", "Physical Billing", "Products", "Orders"],
        "Customer": ["Home", "Shop Online"]
    }
    allowed_pages = role_permissions[role]

    # Validate active page selection against role
    if "nav_selection" not in st.session_state or st.session_state["nav_selection"] not in allowed_pages:
        st.session_state["nav_selection"] = allowed_pages[0]

    st.sidebar.markdown("---")
    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Go to:", allowed_pages, index=allowed_pages.index(st.session_state["nav_selection"]))
    st.session_state["nav_selection"] = choice

    # Routing
    views = {
        "Home": page_home,
        "Products": page_products,
        "Shop Online": page_shop_online,
        "Physical Billing": page_physical_billing,
        "Orders": page_orders,
    }
    views[choice]()


if __name__ == "__main__":
    main()
