# Coffee Machine Simulator

**Author:** Abubakar Iqbal  
**Program:** NAVTTC / NIAI (NETSOL Technologies)  
**Category:** Homework Assignment  

---

## Overview
An object-oriented Python simulator mimicking the operations of a standard automated hot beverage dispensing machine.

---

## Features
1. **Menu & Resource Management**:
   - Offers Espresso, Latte, and Cappuccino with specific water, milk, and coffee bean recipes.
   - Live resource validation before accepting payment.
2. **Coin Processing & Payment**:
   - Processes quarters ($0.25), dimes ($0.10), nickels ($0.05), and pennies ($0.01).
   - Validates sufficient payment and computes accurate change refund.
3. **Admin Diagnostic Reporting**:
   - `report`: Displays remaining resource stocks and total collected money.
   - `off`: Shuts down the machine simulator gracefully.

---

## How to Run
```bash
python coffee_machine_simulator.py
```
