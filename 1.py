from sklearn import datasets

house_price = datasets.load_boston()
print(house_price.data)
digits = datasets.load_digits()
print(digits.images)