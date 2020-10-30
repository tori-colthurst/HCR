import json
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

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
features_list = []
feat_turned = {"1":[], "2":[], "3":[]}
feat_not_turned = {"1":[], "2":[], "3":[]}
label_turning = []
max_cyclists = 1747
total_diff_cyclists = 0

for cyclist in cyclist_data.values():

    # only evaluate every 50 cyclists
    max_cyclists -= 1
    if max_cyclists % 50:
        continue

    # is this cyclist turning?
    turned = -1
    turning_time = (-1,-1)
    for label in cyclist["motion_primitives"]["mp_labels"]:
        l = label["mp_label"]
        if turned == -1 and (l == "tl" or l == "tr"):
            turned = 1
            # capture the time they are turning
            turning_time = (float(label["start_time"]), float(label["end_time"]))

    v_prev = 0
    v_cur = 0
    v_x = {"prev":0, "curr":0}
    v_y = {"prev":0, "curr":0}

    acc = 0
    times = cyclist["trajectory"][0]["Timestamp"]
    x_val = cyclist["trajectory"][0]["x"]
    y_val = cyclist["trajectory"][0]["y"]
    # z_val = cyclist["trajectory"][0]["z"]
    flag_diff_cyclist = 0

    i = 2
    while turned and i < len(x_val)-1:
    # for i in range(0, len(x_val)-1):

        t = float(times[i-2])
        if t < turning_time[0] - 1000000000*3 or t > turning_time[1] + 1000000000*3:
            i += 3
            continue

        flag_diff_cyclist = 1

        feat = []
        dt = (float(times[i])-t)

        x = float(x_val[i-2])
        y = float(y_val[i-2])
        # z = float(z_val[i])

        dx = (float(x_val[i])-x)
        dy = (float(y_val[i])-y)
        # dz = (float(z_val[i+1])-z)

        v_x["curr"] = dx/dt
        v_y["curr"] = dy/dt
        # v_z = dz/dt
        v_cur = (v_x["curr"]**2+v_y["curr"]**2)**0.5

        if i > 2:
            a = abs(v_cur-v_prev)/dt
            a_x = (v_x["curr"]-v_x["prev"])/dt
            a_y = (v_y["curr"]-v_y["prev"])/dt

            # feat = [x,y,dx,dy,v_cur,a]
            feat = [dx,dy,v_x["curr"],v_y["curr"],a_x,a_y]
            features_list.append(feat)

            if t >= turning_time[0] and t <= turning_time[1]:
                label_turning.append(1)
                feat_turned["1"].append(v_x["curr"])
                feat_turned["2"].append(v_y["curr"])
                feat_turned["3"].append(v_x["curr"])
            else:
                label_turning.append(-1)
                if abs(v_y["curr"]) < 0.02:
                    feat_not_turned["1"].append(v_x["curr"])
                    feat_not_turned["2"].append(v_y["curr"])
                    feat_not_turned["3"].append(v_x["curr"])

        v_prev = v_cur
        v_x["prev"] = v_x["curr"]
        v_y["prev"] = v_y["curr"]
        i += 3

    if flag_diff_cyclist:
        total_diff_cyclists += 1


print("TOTAL # DIFF CYCLISTS IN DATA: "+str(total_diff_cyclists))

"""
Plot cyclist data using matplotlib.
"""
plt.figure(figsize=(8,6))
plt.scatter(feat_turned["1"],feat_turned["2"], marker='+',color='green')
plt.scatter(feat_not_turned["1"],feat_not_turned["2"],marker='_',color='red')
plt.xlabel("velocity along the x axis (units unclear)")
plt.ylabel("velocity along the y axis (units unclear)")
plt.title("Velocity y vs. Velocity x")
plt.legend(["Data while turning.", "Data while not turning."])
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(feat_turned["1"], feat_turned["2"], feat_turned["3"], marker='+',color='green')
# ax.scatter(feat_not_turned["1"], feat_not_turned["2"], feat_not_turned["3"], marker='_',color='red')
# plt.show()


"""
Training vs. Testing data split
"""
## Extract the target values / trim input data
X = features_list
Y = label_turning

print(len(X))

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
# print(y_test)

"""
We can use the Scikit learn library and just call the related functions to implement the SVM model.
"""
# clf = SVC(kernel='linear')
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# score = accuracy_score(y_test,y_pred)
# print("ACCURACY: "+str(score))



"""
Reference of SVM implementation from scratch(ish).
"""
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
