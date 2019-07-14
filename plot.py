
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import IPython
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
#import mplleaflet

IPython.get_ipython().run_line_magic('matplotlib', 'inline')

calendar = pd.read_csv('data/calendar.csv')
listings = pd.read_csv('data/listings.csv')
reviews = pd.read_csv('data/reviews.csv')
calendar.head()


#%%
print(len(listings))
listings.head()

#%%
listings.price.head()

#%%
# clean price data
listings.price = listings.price.apply(lambda x: x.split('.')[0]).replace('[^0-9]', '', regex=True).apply(lambda x: int(x)) 

#%%
listings.price.head()

#%%
fig = plt.figure(figsize=(25,25))

m = Basemap(projection='merc', llcrnrlat=42.23, urcrnrlat=42.4, llcrnrlon=-71.18, urcrnrlon=-70.99,)

m.drawcounties()

num_colors = 20
values = listings.price
cm = plt.get_cmap('coolwarm')
scheme = [cm(i / num_colors) for i in range(num_colors)]
bins = np.linspace(values.min(), values.max(), num_colors)
listings['bin'] = np.digitize(values, bins) - 1
cmap = mpl.colors.ListedColormap(scheme)

color = [scheme[listings[(listings.latitude==x)&(listings.longitude==y)]['bin'].values] 
             for x,y in zip(listings.latitude, listings.longitude)]

x,y = m(listings.longitude.values, listings.latitude.values)
scat = m.scatter(x,y, s = listings.price, color = color, cmap=cmap, alpha=0.8)


# Draw color legend.
                        #[left, top, width, height]
ax_legend = fig.add_axes([0.21, 0.12, 0.6, 0.02])
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])


plt.show()

#%% [markdown]
# Listings dataset have several geospatial variables, the most important of which are longitude and latitude, which give the exact coordinates of the BnB in question. So plot every BnB location on a map:

#%%
plt.scatter(listings['longitude'], listings['latitude']);

#%% [markdown]
# For close look of position, I use mplleaflet with undersampling(1000). mplleaflet is a tool that automatically takes a coordinate matplotlib plot of any kind and places it on top of a leaflet slippy map.

#%%

sample = listings.sample(1000)
plt.scatter(sample['longitude'], sample['latitude'])

mplleaflet.display()


#%%
reviews.head()


#%%



