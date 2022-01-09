class Employee:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def change_age(self,x,y):
        self.age=x+y
    
    def change_age_(self):
        self.change_age(self.age,1)

Bill = Employee("Bill", 20)

Bill.change_age(20,10)
Bill.change_age_()
print(Bill.age)

# hi
# hi