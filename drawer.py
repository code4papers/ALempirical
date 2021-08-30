import csv
import matplotlib.pyplot as plt

x_index = [(i + 1) * 0.2 for i in range(10)]

x_index = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"]
first = []
second = []
third = []
with open('results/lstm_random_win500.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        first.append(float(row[0]))


# with open('results/yelp_gru_com_win0.02.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         third.append(float(row[0]))

first_fi = []
second_fi = []
# third_fi = []
for i in range(10):
    first_fi.append((first[i] + first[i + 10] + first[i + 20]) / 3)
    second_fi.append((second[i] + second[i + 10] + second[i + 20]) / 3)
    # third_fi.append((third[i] + third[i + 10] + third[i + 20]) / 3)

plt.plot(x_index, first_fi, label="without diversity")
plt.plot(x_index, second_fi, label="with diversity")
# plt.plot(x_index, third_fi, label="combine")

# plt.plot(x_index, first, label="without diversity")
# plt.plot(x_index, second, label="with diversity")
# plt.plot(x_index, third, label="combine")
plt.xlabel('precentage')
plt.ylabel('acc')
plt.legend()
plt.show()


