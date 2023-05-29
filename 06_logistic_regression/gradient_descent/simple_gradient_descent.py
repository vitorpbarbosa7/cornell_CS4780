


# fw = 5(w-11)**4

# gw = 20(w-11)**3


def fw(wi):
	
	return 5*(w-11)**4

def gw(wi):

	return 20*(wi - 11)**3

def update(wi, eta):

	w = wi - eta*gw(wi)

	return w

eta = 1/40
w0 = 13

values = []
print(f'\n\n eta equals to {eta}')
for i in range(5):
	
	w = update(w0, eta)
	w0 = w
	print(w)
	values.append(w)

print(values)
print(fw(w))


res = fw(w0)
eta = 1/80
w0 = 13

print(f'\n\n eta equals to {eta}')
while res > 0.1:
	
	w = update(w0, eta)
	w0 = w
	res = fw(w0)
	print(w)
	print(res)













