 -- BikeStores
 
-- Inner Join

--Show total sales amount per store.
SELECT s.store_id, s.store_name, SUM(oi.quantity * oi.list_price) AS total_sales
FROM sales.stores s
JOIN sales.orders o ON s.store_id = o.store_id
JOIN sales.order_items oi ON o.order_id = oi.order_id
GROUP BY s.store_id, s.store_name;

-- List all customers who ordered a mountain bike.
SELECT DISTINCT c.customer_id, c.first_name, c.last_name
FROM sales.customers c
JOIN sales.orders o ON c.customer_id = o.customer_id
JOIN sales.order_items oi ON o.order_id = oi.order_id
JOIN production.products p ON oi.product_id = p.product_id
WHERE p.product_name LIKE '%mountain bike%';

--  List customers who placed more than 5 orders.
SELECT c.customer_id, c.first_name, c.last_name, COUNT(o.order_id) AS total_orders
FROM sales.customers c
JOIN sales.orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING COUNT(o.order_id) > 5;

-- List top 3 selling products by quantity.
SELECT TOP 3 p.product_id, p.product_name, SUM(oi.quantity) AS total_quantity_sold
FROM production.products p
JOIN sales.order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name
ORDER BY total_quantity_sold DESC;

-- Show average product price by category.
SELECT c.category_id, c.category_name, AVG(p.list_price) AS avg_price
FROM production.categories c
JOIN production.products p ON c.category_id = p.category_id
GROUP BY c.category_id, c.category_name;

-- Find staff members with highest total sales value.
SELECT st.staff_id, st.first_name, st.last_name, SUM(oi.quantity * oi.list_price) AS total_sales
FROM sales.staffs st
JOIN sales.orders o ON st.staff_id = o.staff_id
JOIN sales.order_items oi ON o.order_id = oi.order_id
GROUP BY st.staff_id, st.first_name, st.last_name
ORDER BY total_sales DESC;

-- Left Join

-- List products that have never been ordered.
SELECT p.product_id, p.product_name
FROM production.products p
LEFT JOIN sales.order_items oi ON p.product_id = oi.product_id
WHERE oi.product_id IS NULL;

-- List all staff members who haven’t processed any orders
SELECT st.staff_id, st.first_name, st.last_name
FROM sales.staffs st
LEFT JOIN sales.orders o ON st.staff_id = o.staff_id
WHERE o.staff_id IS NULL;

-- Subqueries

-- Find the product(s) with the highest list price.
SELECT product_id, product_name, list_price
FROM production.products
WHERE list_price = (
    SELECT MAX(list_price)
    FROM production.products
);

-- Find products where list price is above the average.
SELECT product_id, product_name, list_price
FROM production.products
WHERE list_price > (
    SELECT AVG(list_price) FROM production.products
);

-- Sales Order

-- Inner Join

-- Display all products and their categories.
SELECT 
    P.ProductName,
    C.CategoryDescription
FROM 
    Products P
INNER JOIN 
    Categories C ON P.CategoryID = C.CategoryID;

-- Find all the customers who have ever ordered a bicycle helmet.
SELECT DISTINCT C.CustomerID, C.CustFirstName
FROM Customers C
JOIN Orders O
    ON C.CustomerID = O.CustomerID
JOIN Order_Details OD
    ON O.OrderNumber = OD.OrderNumber
JOIN Products P
    ON OD.ProductNumber = P.ProductNumber
WHERE P.ProductName LIKE '%helmet%';

-- Left Outer Join

-- What products have never been ordered.
SELECT P.ProductNumber, P.ProductName
FROM Products P
LEFT JOIN Order_Details OD
    ON P.ProductNumber = OD.ProductNumber
WHERE OD.ProductNumber IS NULL;

-- Union

-- Build a single mailing list that consist of name, address, city, state and ZIP code of customers and Vendors.
SELECT 
    CONCAT(CustFirstName, ' ', CustLastName) AS Name,
    CustStreetAddress AS Address,
    CustCity AS City,
    CustState AS State,
    CustZipCode AS ZipCode
FROM Customers

UNION

SELECT 
    VendName AS Name,
    VendStreetAddress AS Address,
    VendCity AS City,
    VendState AS State,
    VendZipCode AS ZipCode
FROM Vendors;

-- Show me all the customers and employee names and addresses including any duplicates, sorted by ZIP Code.
SELECT 
    CustFirstName + ' ' + CustLastName AS FullName,
    CustStreetAddress AS StreetAddress,
    CustCity AS City,
    CustState AS State,
    CustZipCode AS ZipCode
FROM Customers

UNION ALL

SELECT 
    EmpFirstName + ' ' + EmpLastName AS FullName,
    EmpStreetAddress AS StreetAddress,
    EmpCity AS City,
    EmpState AS State,
    EmpZipCode AS ZipCode
FROM Employees

ORDER BY ZipCode;

-- Subqueries

-- List vendors and a count of products they sell to us.
SELECT VendName,
(SELECT COUNT(*)
FROM Product_Vendors
WHERE Vendors.VendorID = Product_Vendors.VendorID) As ProductsCount
FROM Vendors;

-- Aggregate

-- How many customers we have in the state of California?
SELECT COUNT(*) AS NoOfCustomers
FROM Customers
WHERE CustState = 'CA';

/*List the product names and numbers that have a quoted price greater than or equal to the overall average retail 
price in the products table.*/
SELECT ProductNumber, ProductName, RetailPrice
FROM Products
WHERE RetailPrice >= (
    SELECT AVG(RetailPrice)
    FROM Products
);

-- School Scheduling

-- Left Outer Join

-- List the faculty members not teaching a class.
SELECT F.StaffID, F.Title
FROM Faculty F
LEFT JOIN Faculty_Classes FC ON F.StaffID = FC.StaffID
WHERE FC.ClassID IS NULL;

SELECT * FROM Faculty_Classes
WHERE StaffID=98010;

-- Aggregate

-- What is the largest salary we pay to any staff member?
SELECT StaffID, StfFirstName, Salary
FROM Staff
WHERE Salary = (
    SELECT MAX(Salary)
    FROM Staff
);
-- To Check for Duplicates
SELECT * FROM Staff;

-- Bowling League

-- Left Outer Join

-- Show me tournaments that have not been played yet.
SELECT Tournaments.TourneyID, Tourney_matches.MatchID
FROM Tournaments
LEFT JOIN Tourney_Matches 
  ON Tournaments.TourneyID = Tourney_matches.TourneyID
WHERE Tourney_matches.MatchID IS NULL;

-- Subqueries

-- Display the bowlers and their high score.
USE BowlingLeague;

SELECT 
    BowlerFirstName + ' ' + BowlerLastName AS BowlerName,
    (
        SELECT MAX(RawScore)
        FROM Bowler_Scores
        WHERE Bowler_Scores.BowlerID = Bowlers.BowlerID
    ) AS HighScore
FROM Bowlers;

-- Entertainment Agency

-- Inner Join

-- Show me entertainers, the start and end dates of their contracts and the contract price
SELECT 
    Entertainers.EntStageName, 
    Engagements.StartDate, 
    Engagements.EndDate, 
    Engagements.ContractPrice
FROM 
    Entertainers
INNER JOIN 
    Engagements 
ON 
    Entertainers.EntertainerID = Engagements.EntertainerID;

-- Find the entertainers who played engagements for customers Berg or Hallmark
SELECT 
    Entertainers.EntStageName, 
    Engagements.CustomerID,
    Customers.CustFirstName, 
    Customers.CustLastName
FROM 
    Entertainers
INNER JOIN 
    Engagements ON Entertainers.EntertainerID = Engagements.EntertainerID
INNER JOIN 
    Customers ON Engagements.CustomerID = Customers.CustomerID
WHERE 
    Customers.CustLastName IN ('Hallmark', 'Berg');

-- Left Outer Join

--List entertainers who have never been booked.
SELECT 
    Entertainers.EntertainerID, 
    Entertainers.EntStageName, 
    Engagements.EngagementNumber
FROM 
    Entertainers
LEFT OUTER JOIN 
    Engagements 
ON 
    Entertainers.EntertainerID = Engagements.EntertainerID
WHERE 
    Engagements.EngagementNumber IS NULL;

-- Subqueries

--Display all customers and the date of the last booking each customer made.
USE EntertainmentAgency;

SELECT Concat(CustFirstName, ' ' , CustLastName) As CustomerName,
(SELECT MAX(StartDate)
FROM Engagements
WHERE Customers.CustomerID = Engagements.CustomerID) As LastBookingDate
FROM Customers;

-- Aggregate

-- What was the total value of all engagements booked in October 2017.
USE EntertainmentAgency;

SELECT SUM(ContractPrice) AS TotalValue
FROM Engagements
WHERE TRY_CONVERT(date, StartDate, 103) 
      BETWEEN '2017-10-01' AND '2017-10-31';
/*
- TRY_CONVERT(date, StartDate, 103):
- Converts the StartDate from dd/mm/yyyy format.
- 103 is the British/French style date format (dd/mm/yyyy).
- BETWEEN '2017-10-01' AND '2017-10-31':
- Now operates on valid date types without breaking.*/

SELECT SUM(ContractPrice) AS TotalValue
FROM Engagements
WHERE FORMAT(TRY_CONVERT(date, StartDate, 103), 'yyyy-MM') = '2017-10';

SELECT SUM(ContractPrice) AS TotalValue
FROM Engagements
WHERE DATEPART(YEAR, TRY_CONVERT(date, StartDate, 103)) = 2017
  AND DATEPART(MONTH, TRY_CONVERT(date, StartDate, 103)) = 10;

SELECT SUM(ContractPrice) AS TotalValue
FROM Engagements
WHERE 
    TRY_CONVERT(date, StartDate, 103) >= '2017-10-01'
    AND TRY_CONVERT(date, StartDate, 103) <  '2017-11-01';

-- Recipes

-- Inner Join

-- Show me the main course recipes and list all the ingredients.
SELECT 
    Recipe_Classes.RecipeClassDescription, 
    Recipes.RecipeID, 
    recipe_ingredients.IngredientID, 
    Ingredients.IngredientName
FROM Recipe_Classes
INNER JOIN Recipes 
    ON Recipe_Classes.RecipeClassID = Recipes.RecipeClassID
INNER JOIN recipe_ingredients 
    ON Recipes.RecipeID = recipe_ingredients.RecipeID
INNER JOIN Ingredients 
    ON recipe_ingredients.IngredientID = Ingredients.IngredientID
WHERE Recipe_Classes.RecipeClassDescription = 'Main Course';

-- Left Outer Join

-- List ingredients not used in any recipe yet.
SELECT 
    Ingredients.IngredientID, 
    Ingredients.IngredientName, 
    Recipe_Ingredients.RecipeID
FROM Ingredients
LEFT JOIN Recipe_Ingredients 
    ON Ingredients.IngredientID = Recipe_Ingredients.IngredientID
WHERE Recipe_Ingredients.RecipeID IS NULL;