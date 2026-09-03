class CoffeeMachine:
    def __init__(self):
        # 1. Initialize the machine's resources and money
        self.resources = {
            "water": 300,
            "milk": 200,
            "coffee": 100,
        }
        self.money = 0
        
        # 2. Define the menu with ingredient requirements and costs
        self.menu = {
            "espresso": {"water": 50, "milk": 0, "coffee": 18, "cost": 1.5},
            "latte": {"water": 200, "milk": 150, "coffee": 24, "cost": 2.5},
            "cappuccino": {"water": 250, "milk": 100, "coffee": 24, "cost": 3.0}
        }
    def report(self):
        print(f"water:{self.resources["water"]}ml")
        print(f"milk:{self.resources["milk"]}ml")
        print(f"coffee:{self.resources["coffee"]}g")
        print(f"Money:{self.money:.2f}")
    

        
