
predictions = open("predictions.csv", "r").readlines()
target = open("test.csv").readlines()


MAPE = 0.0

if len(predictions) != len(target):
    print("line number not the same")
else:
    for i in range(1, len(predictions)):
        MAPE += abs((float(predictions[i]) - float(target[i])) / float(target[i]))

MAPE /= (len(predictions) - 1)
print(f"MAPE: {MAPE}")
