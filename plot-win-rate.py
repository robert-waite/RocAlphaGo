import json
import matplotlib.pyplot as plt

with open('metadata.json') as data_file:
    metadata = json.load(data_file)



plt.grid()
#plt.xlim([0, 500])
#plt.ylim([.10, .90])

wr_dict = metadata["win_ratio"]
winrates = []
for key, value in sorted(wr_dict.items()):
	winrates.append(value[1])


x_vals = []
y_vals = []
for data, iteration in zip(winrates, xrange(len(winrates))):
	x_vals.append(data)
	y_vals.append(iteration)


plt.plot(y_vals, x_vals)
plt.show()