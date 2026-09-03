# 🛒 Al-Fatah Super Store Management System

A beginner-friendly, modular Super Store Management and Point of Sale (POS) system built using **Python**, **Streamlit**, and **Pandas**. 

It uses CSV files directly as the database—no SQL or external server configuration required.

---

## 📁 Project Structure

```text
alfatah_store/
├── app.py              # Streamlit Web UI (Presentation Layer)
├── database.py         # Direct CSV Data Access & Atomic Checkout (Data Layer)
├── README.md           # Project documentation and setup guide
└── data/
    ├── products.csv    # Inventory database (22 products across 8 categories)
    └── orders.csv      # Order transaction log
```

---

## ✨ Key Features

1. **Role-Based Access Control (RBAC)**:
   - **Admin**: Full access to all modules, including product catalog CRUD (Add, Edit price/stock, Delete), low stock alerts, and financial analytics.
   - **Cashier (Employee)**: Access to Physical Store Billing (POS), view-only catalog, and order history.
   - **Customer**: Clean shopping experience restricted to Store Home and Shop Online.

2. **Multi-Item Shopping Cart**:
   - Add multiple items with custom quantities to an in-memory cart before checking out.
   - Dynamic real-time cart table with subtotals and grand total calculation.

3. **Atomic Transactions & Stock Validation**:
   - Validates live stock across **all** items in the cart before finalizing.
   - Prevents stock deductions and order creation if any single item lacks sufficient inventory.
   - Updates `products.csv` and `orders.csv` in tandem upon success.

4. **Real-time Low Stock Warning**:
   - Automatically scans and displays warning banners on Admin/Cashier views if any product drops below 10 units.

5. **Counter Receipt Generation**:
   - Formats and displays an itemized ASCII receipt upon in-store checkout completion.

6. **Orders & Revenue Analytics**:
   - Filter order history by channel (`All`, `online`, `physical`).
   - Summary cards displaying total transactions, gross revenue, and total units sold.

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python installed, along with `pandas` and `streamlit`:
```bash
pip install pandas streamlit
```

### 2. Run the Application
Open your terminal, navigate into the project directory, and launch Streamlit:
```bash
cd "/home/bakar/Desktop/Python course/NAVTTC/alfatah_store"
streamlit run app.py
```

---

## 📊 CSV Schemas

### `data/products.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `product_id` | Integer | Unique identifier for product |
| `name` | String | Product title |
| `category` | String | Department (Grocery, Dairy, Bakery, etc.) |
| `price` | Integer | Price in PKR |
| `stock` | Integer | Available units in inventory |

### `data/orders.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `order_id` | Integer | Sequential transaction ID |
| `product_name` | String | Product name purchased |
| `quantity` | Integer | Units ordered |
| `total_price` | Integer | Calculated amount (Price × Quantity) |
| `date` | String | Timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `channel` | String | Order source (`online` or `physical`) |

---

## 💡 CRUD Operations Map

- **Create**: Add new product in `Products` page (Admin only).
- **Read**: Product listing and orders table using `st.dataframe()`.
- **Update**: Adjust price and stock in `Products` page (Admin only).
- **Delete**: Remove obsolete products in `Products` page (Admin only).
