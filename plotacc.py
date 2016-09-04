import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open('metadata.json') as data_file:
    phasedrmsprop = json.load(data_file)

with open('metadataContinue.json') as data_file:
    continued = json.load(data_file)

with open('metadataphase1.json') as data_file:
    phaseddata1 = json.load(data_file)

with open('metadataphase2.json') as data_file:
    phaseddata2 = json.load(data_file)

with open('metadataAdamV2.json') as data_file:
    adamv2data = json.load(data_file)

with open('metadataROCPARAMS.json') as data_file:
    data = json.load(data_file)

with open('metadataCOMM.json') as data_file:
    rocdata = json.load(data_file)

def gen_graph_data(dataset, acc_data, val_acc_data, y_labels, epoch_length, offset=0):
    if False:
        metric = "loss"
        metric_val = "val_loss"
        if (offset == 0):
            acc_data.append(10.0)
            val_acc_data.append(10.0)
            y_labels.append(10.0)
    else:
        metric = "acc"
        metric_val = "val_acc"
        if(offset == 0):
            acc_data.append(0.0)
            val_acc_data.append(0.0)
            y_labels.append(0.0)

    for epoch, iteration in zip(dataset["epochs"], xrange(len(dataset["epochs"]))):
        acc_data.append(epoch[metric])
        val_acc_data.append(epoch[metric_val])
        y_labels.append((iteration + 1) * epoch_length + offset)

acc = []
val_acc = []
y_labels = []
gen_graph_data(data, acc, val_acc, y_labels, 500000)

roc_acc = []
roc_val_acc = []
roc_y_labels = []
gen_graph_data(rocdata, roc_acc, roc_val_acc, roc_y_labels, 40000)

adamv2_acc = []
adamv2_val_acc = []
adamv2_y_labels = []
gen_graph_data(adamv2data, adamv2_acc, adamv2_val_acc, adamv2_y_labels, 250000)

phased1_acc = []
phased1_val_acc = []
phased1_y_labels = []
gen_graph_data(phaseddata1, phased1_acc, phased1_val_acc, phased1_y_labels, 250000)

phased2_acc = []
phased2_val_acc = []
phased2_y_labels = []
gen_graph_data(phaseddata2, phased2_acc, phased2_val_acc, phased2_y_labels, 250000)

continued_acc = []
continued_val_acc = []
continued_y_labels = []
gen_graph_data(continued, continued_acc, continued_val_acc, continued_y_labels, 250000, 29600000)

phasedrmsprop_acc = []
phasedrmsprop_val_acc = []
phasedrmsprop_y_labels = []
gen_graph_data(phasedrmsprop, phasedrmsprop_acc, phasedrmsprop_val_acc, phasedrmsprop_y_labels, 250000)


mil_to_show = 60
plt.grid()
plt.xlim([0, mil_to_show * 1000000])
plt.ylim([.40, .50])
#plt.ylim([.40, .51])
#plt.ylim([.46, .485])
#plt.ylim([1.8, 3])
plt.xticks(range(0, mil_to_show * 1000000, 1000000), range(0, mil_to_show))
#plt.yticks(range(1, 20), range(1, 20))
plt.plot(y_labels, acc, color='blue')
plt.plot(y_labels, val_acc, color='green')
plt.plot(roc_y_labels, roc_acc, color='red', alpha=0.25)
plt.plot(roc_y_labels, roc_val_acc, color='cyan', alpha =0.75)
plt.plot(phased1_y_labels, phased1_acc, color='blue')
plt.plot(phased1_y_labels, phased1_val_acc, color='green')
plt.plot(phased2_y_labels, phased2_acc, color='blue')
plt.plot(phased2_y_labels, phased2_val_acc, color='green')
plt.plot(continued_y_labels, continued_acc, color='blue')
plt.plot(continued_y_labels, continued_val_acc, color='green')
plt.plot(adamv2_y_labels, adamv2_acc, color='limegreen')
plt.plot(adamv2_y_labels, adamv2_val_acc, color='orange')
plt.plot(phasedrmsprop_y_labels, phasedrmsprop_acc, color='black')
plt.plot(phasedrmsprop_y_labels, phasedrmsprop_val_acc, color='magenta')

#x = np.arange(0, 5440000000, 250000)
#y = 55.496377030690375770369540667998 * np.log(10055.650654686864088316399448351 * x)
#plt.plot(x, y)



plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('state-action pairs processed (in millions)')
blue_patch = mpatches.Patch(color='blue', label='Sym with Phase Training')
green_patch = mpatches.Patch(color='green', label='Sym with Phase Validation')
red_patch = mpatches.Patch(color='red', alpha=0.25, label='Wrongu Run Training')
cyan_patch = mpatches.Patch(color='cyan', alpha =0.50, label='Wrongu Run Validation')
limegreen_patch = mpatches.Patch(color='limegreen', label='Adam Training')
orange_patch = mpatches.Patch(color='orange', label='Adam Validation')
black_patch = mpatches.Patch(color='black', label='Phased Training')
magenta_patch = mpatches.Patch(color='magenta', label='Phased Validation')
#plt.legend(handles=[blue_patch, green_patch, red_patch, cyan_patch, black_patch, magenta_patch], loc='lower right')
#plt.legend(['sgd-train', 'sgd-validation', 'wrongu-sgd-train', 'wrongu-sgd-validation', 'adam-train', 'adam-validation'], loc='upper left')
plt.show()
# # summarize history for loss
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()