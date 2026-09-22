import pandas as pd
import os
class StudentManagement:
    def __init__(self,filename="student.csv"):
        self.filename=filename
        if not os.path.exists(self.filename):
            df=pd.DataFrame(columns=["Roll","Name"])
            df.to_csv(self.filename,index=False)
    def add_student(self):
        rollno=input("Enter Roll :")
        name=input("Enter Name : ")
        
        df=pd.read_csv(self.filename)
        if rollno in df["Roll"].astype(str).values:
            print("Roll number already exist")
            return
        df=pd.concat([df,pd.DataFrame([{"Roll":rollno,"Name":name}])],ignore_index=True)
        df.to_csv(self.filename,index=False)
        print("Student Added !")
        
    def view_student(self):
        df=pd.read_csv(self.filename)
        if df.empty:
            print("No Student record found")
        else:
            print("\n..Student Records..")
            print(df.to_string(index=False))
            
    def search_student(self):
        rollno=input("Enter roll no to search : ")
        df=pd.read_csv(self.filename)
        result=df[df["Roll"].astype(str)==rollno]
        if result.empty:
            print("Student not found")
        else:
            print("\n Student Found")
            print(result.to_string(index=False))
    def update_student(self):
        rollno=input("Enter rollno to update : ")
        df=pd.read_csv(self.filename)
        if rollno not in df["Roll"].astype(str).values:
            print("Student not found")
            return
        name=input("Enter New Name :")
        df.loc[df["Roll"].astype(str)==rollno,"Name"]=name
        df.to_csv(self.filename,index=False)
        print("Student updated")
        
    def delete_student(self):
        rollno=input("Enter Roll no To Delete : ")
        df=pd.read_csv(self.filename)
        if rollno not in df["Roll"].astype(str).values:
            print("Student not found")
            return
        else:
            df=df[df["Roll"].astype(str)!=rollno]
            df.to_csv(self.filename,index=False)
            print("student deleted ")
    def menu(self):
        while True:
            print("\n STUDENT MANAGEMENT SYSTEM ")
            print("1. Add Student")
            print("2. View Students")
            print("3. Search Student")
            print("4. Update Student")
            print("5. Delete Student")
            print("6. Exit")
            
            choice=input("Enter your choice : ")
            if choice =="1":
                self.add_student()
            elif choice=="2":
                self.view_student()
            elif choice=="3":
                self.search_student()
            elif choice=="4":
                self.update_student()
            elif choice=="5":
                self.delete_student()
            elif choice=="6":
                print("Exiting")
                break
            else:
                print("Invalid choice")
if __name__=="__main__":
    sms=StudentManagement()
    sms.menu()