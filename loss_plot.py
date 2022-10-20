import csv

csv_file_name = './train_log_cornet_20221018_0.csv'
epoch_list = []
train_loss_list = []
train_mae_list = []
val_loss_list = []
val_mae_list = []
with open(csv_file_name) as f:
    items = csv.reader(f)
    for i, item in enumerate(items):
        epoch = item[0]
        train_loss = item[1]
        train_mae = item[2]
        val_loss = item[3]
        val_mae = item[4]
        if i > 0:
            epoch_list.append(eval(epoch))
            train_loss_list.append(eval(train_loss))
            train_mae_list.append(eval(train_mae))
            val_loss_list.append(eval(val_loss))
            val_mae_list.append(eval(val_mae))
#     print(val_loss_list)

import matplotlib.pyplot as plt
plt.figure()
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epoch')
plt.ylabel('loss')

# print(val_loss_list)

plt.plot(epoch_list, val_loss_list, color='red', linewidth=1, linestyle="solid", label="val loss")
plt.legend()
plt.plot(epoch_list, train_loss_list, color='blue', linewidth=1, linestyle="solid", label="train loss")
plt.legend()
plt.title('Loss curve')
plt.show()
