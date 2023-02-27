import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
font_set=FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
def drawPlot(list,epoch,xlabel,ylabel,title):
    plt.figure()
    x1=range(1,epoch+1)
    y1=list
    plt.cla()
    plt.title(title, fontproperties=font_set)
    plt.plot(x1,y1,'.-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(f"./{ylabel}-{xlabel}.png")
    plt.show()

drawPlot([54,34,23,10,3,2,1],7,"epochs","acc","精度")