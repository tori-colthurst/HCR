import json
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

"""
Grab cyclist trajectory data from dataset json files. Total of 1746 cyclists.
Variable - cyclist_data - contains trajectory info for every cyclist, tagged 0-1745.

Data format is as follows:
{"vru_type": "bike",
  "trajectory2: [{"Timestamp": [LIST OF UTC TIMESTAMPS],
                           "x": [LIST OF X POSISTIONS],
                           "y": [LIST OF Y POSISTIONS],
                           "z": [LIST OF Z POSISTIONS],
                           "x_smoothed": [LIST OF SMOOTHED X POSISTIONS],
                           "y_smoothed": [LIST OF SMOOTHED Y POSISTIONS],
                           "z_smoothed": [LIST OF SMOOTHED Z POSISTIONS]}],
  "motion_primitives": {"mp_labels":
          [{"mp_label": LABEL NAME,
             "start_time": START UTC TIMESTAMP,
             "end_time": END UTC TIMESTAMP, ...}, ...]}}
"""
files = list(os.walk("./extended_cyclist_dataset"))[0][2]
tag = 0
cyclist_data = {}
for filename in files:
    # print(filename)
    with open('./extended_cyclist_dataset/'+filename) as f:
        cyclist_data[tag] = json.load(f)
        tag += 1

"""
Features: current position (x,y,z), delta from old position (dx,dy,dz), current velocity (v), current acceleration (a)
Labels: {'', 'foot_off_ground', 'grabbing_handle_right', 'releasing_handle_right', 'wait', 'tl', 'shoulder_check_right', 'start', 'hand_signal_left', 'releasing_handle_left', 'grabbing_handle_left', 'move', 'hand_signal_right', 'starting_movement', 'shoulder_check_left', 'pedaling', 'stop', 'tr', 'out_of_saddle', 'foot_on_ground'}
"""
avg_acc = []
avg_v = []
avg_acc_r = []
avg_v_r = []
features_list = []
label_turning = []
label_signaling = []
counting_cyclists = 1747#int(1746/2)
for cyclist in cyclist_data.values():
    counting_cyclists -= 1
    if counting_cyclists % 20:
        # print(counting_cyclists)
        continue
    raised = -1
    raising_time = (-1,-1)
    turned = -1
    turning_time = (-1,-1)
    for label in cyclist["motion_primitives"]["mp_labels"]:
        l = label["mp_label"]
        if raised ==-1 and (l == "hand_signal_left" or l == "hand_signal_right"):
            raised = 1
            raising_time = (float(label["start_time"]), float(label["end_time"]))
        if turned == -1 and (l == "tl" or l == "tr"):
            turned = 1
            turning_time = (float(label["start_time"]), float(label["end_time"]))

    # print(turning_time)

    v = 0
    v_prev = 0
    v_cur = 0
    acc = 0
    times = cyclist["trajectory"][0]["Timestamp"]
    x_val = cyclist["trajectory"][0]["x"]
    y_val = cyclist["trajectory"][0]["y"]
    z_val = cyclist["trajectory"][0]["z"]
    count = 0
    for i in range(0, len(x_val)-1):
        t = float(times[i])
        # print(t)
        if not turned or t < turning_time[0] - 10000000 or t > turning_time[1] + 10000000:
            break
        feat = []
        dt = (float(times[i+1])-t)
        x = float(x_val[i])
        y = float(y_val[i])
        z = float(z_val[i])
        dx = (float(x_val[i+1])-x)
        dy = (float(y_val[i+1])-y)
        dz = (float(z_val[i+1])-z)
        v_x = dx/dt
        v_y = dy/dt
        v_z = dz/dt
        v_cur = (v_x**2+v_y**2+v_z**2)**0.5
        # v += v_cur
        if i > 0:
            a = abs(v_cur-v_prev)/dt
            # acc += a
            # count += 1
            feat = [x,y,z,dx,dy,dz,v_cur,a]
            features_list.append(feat)
            if t >= turning_time[0] and t <= turning_time[1]:
                label_turning.append(1)
            else:
                label_turning.append(-1)
            label_signaling.append(raised)
        v_prev = v_cur

    # v = v*100/(count + 1)
    # acc = acc*100000/count
    # if raised:
    #     avg_acc_r.append(acc)
    #     avg_v_r.append(v)
    # else:
    #     avg_acc.append(acc)
    #     avg_v.append(v)
    #

"""
Plot cyclist data using matplotlib.
"""
# plt.figure(figsize=(8,6))
# plt.scatter(avg_v,avg_acc,marker='+',color='green')
# plt.scatter(avg_v_r,avg_acc_r,marker='_',color='red')
# plt.show()

"""
Training vs. Testing data split
"""
## Extract the target values
# df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
# Y = []
# target = df['Species']
# for val in target:
#     if(val == 'Iris-setosa'):
#         Y.append(-1)
#     else:
#         Y.append(1)
# df = df.drop(['Species'],axis=1)
# X = df.values.tolist()
X = features_list
Y = label_turning
print(len(X))
# print(len(Y))
## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(len(x_test))
print(y_test)

# y_train = y_train.reshape(90,1)
# y_test = y_test.reshape(10,1)

"""
We can use the Scikit learn library and just call the related functions to implement the SVM model.
"""
clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
score = accuracy_score(y_test,y_pred)
print("ACCURACY: "+str(score))


def SVM_whole():
    """
    Support Vector Machine
    """
    train_f1 = x_train[:,0]
    train_f2 = x_train[:,1]

    train_f1 = train_f1.reshape(90,1)
    train_f2 = train_f2.reshape(90,1)

    w1 = np.zeros((90,1))
    w2 = np.zeros((90,1))

    epochs = 1
    alpha = 0.0001

    while(epochs < 10000):
        y = w1 * train_f1 + w2 * train_f2
        prod = y * y_train
        print(epochs)
        count = 0
        for val in prod:
            if(val >= 1):
                cost = 0
                w1 = w1 - alpha * (2 * 1/epochs * w1)
                w2 = w2 - alpha * (2 * 1/epochs * w2)

            else:
                cost = 1 - val
                w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
                w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1/epochs * w2)
            count += 1
        epochs += 1

    """
    We now clip the weights as the test data contains only 10 data points.
    We extract the features from the test data and predict the values.
    We obtain the predictions and compare it with the actual values and print the accuracy of our model.
    """
    ## Clip the weights
    index = list(range(10,90))
    w1 = np.delete(w1,index)
    w2 = np.delete(w2,index)

    w1 = w1.reshape(10,1)
    w2 = w2.reshape(10,1)
    ## Extract the test data features
    test_f1 = x_test[:,0]
    test_f2 = x_test[:,1]

    test_f1 = test_f1.reshape(10,1)
    test_f2 = test_f2.reshape(10,1)
    ## Predict
    y_pred = w1 * test_f1 + w2 * test_f2
    predictions = []
    for val in y_pred:
        if(val > 1):
            predictions.append(1)
        else:
            predictions.append(-1)

    print(accuracy_score(y_test,predictions))
