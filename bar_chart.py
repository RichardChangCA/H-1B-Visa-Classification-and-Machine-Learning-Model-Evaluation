import matplotlib
import matplotlib.pyplot as plt

y = [0.81,0.65,0.68,0.45,0.79,0.80,0.80,0.76]


x = ["DT",
"LE",
"KNN",
"GNB",
"SR",
"RF",
"AD",
"VT"]

plt.bar(x, y)
plt.xlabel('Model')
plt.ylabel('Average accuracy')
plt.title(r'Average accuracy with different classifiers')

# Tweak spacing to prevent clipping of ylabel
# fig.tight_layout()
plt.savefig("average_accuracy.png")
plt.show()