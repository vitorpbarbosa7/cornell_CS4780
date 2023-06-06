
def add(x):
	def add_y(y):
		return x + y
	
	return add_y

add_5 = add(5)

result = add_5(3)

print(result)
