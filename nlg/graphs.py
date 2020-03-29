import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('employee.csv')
x = []
y = []
columns = data.columns
print(columns)

pairs = [['Age', 'Attrition'],
         ['BusinessTravel', 'DailyRate'],
         ['EducationField', 'EmployeeNumber'],
         ['DistanceFromHome', 'YearsInCurrentRole']]

x = []
y = []
for pair in pairs:
    colx = pair[0]
    coly = pair[1]
    x.append(data[colx])
    y.append(data[coly])

final_x = []
final_y = []
for i in range(len(pairs)):
    val_x = []
    val_y = []
    for j in range(x[i].shape[0]):
        val_x.append(x[i][j])
        val_y.append(y[i][j])
    final_x.append(val_x)
    final_y.append(val_y)


def plots(x, y, pairs):

    n_rows = int(len(pairs)/2)
    n_cols = int(len(pairs)/2)
    fig, a = plt.subplots(n_rows,n_cols)
    counter = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if counter < len(pairs):
                a[i][j].scatter(x[counter],y[counter])
                a[i][j].set_title(pairs[counter][0] + ' vs ' + pairs[counter][1])
                counter += 1
    plt.show()


plots(final_x, final_y, pairs)