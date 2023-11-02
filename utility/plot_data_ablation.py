import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

plt.rcParams.update({ "text.usetex":True, "font.family": "serif"})

# CUB
axx = [150/150*100, 125/150*100, 100/150*100, 75/150*100, 50/150*100, 37/150*100, 25/150*100, 12/150*100]

fs = 14
zslc = 'mediumblue' # 'chocolate'
gzslc ='mediumblue'
tickcl=12


### ZSL
fig,ax = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(6)

a1, = ax.plot(axx,
        [60.9, 57.4, 51.0, 44.5, 37.4, 31.5, 29.5, 14.3],
        color=zslc, 
        marker="*",
        label="Full ICIS (+ include unseen)")

b1, = ax.plot(axx,
        [60.2, 57.8, 50.4, 44.7, 34.8, 29.3, 22.8, 11.1],
        color=zslc, 
        marker="x",
        linestyle="dashed",
        label="+ Cross-modal")

c1, = ax.plot(axx,
        [58.1, 56.0, 48.7, 44.3, 24.4, 27.6, 21.0, 9.9],
        color=zslc, 
        marker="d",
        linestyle="dashdot",
        label="+ Single-modal")

d1, = ax.plot(axx,
        [54.1, 49.9, 42.8, 34.3,  5.5, 21.8, 20.2, 8.9],
        color=zslc, 
        marker="s",
        linestyle=(0, (5, 10)),
        label="+ Cosine loss")

e1, = ax.plot(axx,
        [41.5, 38.8, 32.6, 28.1,  5.2,  8.0, 13.0, 8.5],
        color=zslc, 
        marker="o",
        linestyle="dotted",
        label="MLP base model")

# set x-axis label
ax.set_xlabel("\% of seen classes", fontsize = fs)
# set y-axis label
ax.set_ylim(bottom=0, top=65)
ax.yaxis.label.set_color(zslc)
ax.set_ylabel("I-ZSL, Acc\%",
              color=zslc,
              fontsize=fs)
ax.tick_params(axis='y', colors=zslc,labelsize=tickcl)
ax.set_xlim(left=axx[-1]+1, right=axx[0]+1)
ax.tick_params(axis='x', labelsize=tickcl)
plt.xticks(np.arange(10, 100, 10))

l = plt.legend([e1, d1, c1, b1, a1], ['MLP base model', '+ Cosine loss', '+ Single-modal', '+ Cross-modal', 'ICIS (full)'],
                 handlelength=3, borderpad=0.7, labelspacing=0.7, loc='lower right', fontsize=8) # 'upper left'

#save the plot as a file
fig.savefig('numsamples_ablation_zsl.pdf',
            format='pdf',
            dpi=1200,
            bbox_inches='tight')

### GZSL
fig,ax = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(6)
# make a plot
a1, = ax.plot(axx,
        [56.7, 54.8, 50.4, 46.1, 38.4, 35.3, 29.5, 17.1],
        color=gzslc, 
        marker="*",
        label="Full ICIS (+ include unseen)")

b1, = ax.plot(axx,
        [56.0, 53.8, 47.7, 44.2, 30.9, 29.7, 23.2, 12.1],
        color=gzslc, 
        marker="x",
        linestyle="dashed",
        label="+ Cross-modal")

c1, = ax.plot(axx,
        [52.7, 51.6, 44.9, 42.6, 14.8, 25.4, 19.7, 10.1],
        color=gzslc, 
        marker="d",
        linestyle="dashdot",
        label="+ Single-modal")

d1, = ax.plot(axx,
        [50.9, 48.3, 42.8, 36.2,  0.0, 23.5, 23.3, 9.8],
        color=gzslc, 
        marker="s",
        linestyle=(0, (5, 10)),
        label="+ Cosine loss")

e1, = ax.plot(axx,
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
        color=gzslc, 
        marker="o",
        linestyle="dotted",
        label="MLP base model")

# set x-axis label
ax.set_xlabel("\% of seen classes", fontsize = fs)
# set y-axis label
ax.set_ylim(bottom=0, top=65)
ax.yaxis.label.set_color(gzslc)
ax.set_ylabel("I-GZSL, H",
              color=gzslc,
              fontsize=fs)
ax.tick_params(axis='y', colors=gzslc,labelsize=tickcl)
ax.set_xlim(left=axx[-1]+1, right=axx[0]+1)
ax.tick_params(axis='x', labelsize=tickcl)
plt.xticks(np.arange(10, 100, 10))

l = plt.legend([e1, d1, c1, b1, a1], ['MLP base model', '+ Cosine loss', '+ Single-modal', '+ Cross-modal', 'ICIS (full)'],
                 handlelength=3, borderpad=0.7, labelspacing=0.7, loc='lower right', fontsize=8)

#save the plot as a file
fig.savefig('numsamples_ablation_gzsl.pdf',
            format='pdf',
            dpi=1200,
            bbox_inches='tight')
