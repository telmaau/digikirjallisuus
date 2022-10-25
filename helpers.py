from importlib import reload
#import digikirjallisuus.saffine
#reload(saffine)
import numpy as np
import re
import digikirjallisuus.saffine.multi_detrending as md
import digikirjallisuus.saffine.detrending_method as dm
from scipy.stats import norm
import matplotlib.pyplot as plt


def integrate(x):
    return np.mat(np.cumsum(x) - np.mean(x))

def normalize(ts, scl01 = False):
    ts01 = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
    ts11 = 2 * ts01 -1
    if scl01:
        return ts01
    else:
        return ts11
    
def optimal_bin(y):
    """ optimal number of bins for histogram
    src: https://academic.oup.com/biomet/article-abstract/66/3/605/232642
    """
    R = max(y)-min(y)
    n = len(y)
    sigma = np.std(y)
    return int(round((R * (n**(1./3.))) / (3.49 * sigma)))

def raw_smooth_plot(story_arc, Hurst):
    y = integrate(story_arc)
    uneven = y.shape[1]%2
    if uneven:
        y = y[0,:-1]

    # afa
    #n = 500
    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)
    ## slope
    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    p = np.poly1d(np.polyfit(x, y, order))
    xp = np.linspace(0, len(x), len(x))

    #fig, ax = plt.subplots(2,1)

    plt.figure(figsize=(12,20))
    plt.subplot(311)
    X = np.mat([float(x) for x in story_arc])
    plt.plot(X.T,'-k', label = 'story arc')
    n = len(story_arc)
    w = int(4 * np.floor(n/20) + 1)

    # format
    for i in range(2,5):
        try:
            _, trend_ww_1 = dm.detrending_method(X, w, i)
            plt.plot(normalize(trend_ww_1).T, label = "$m = {}$".format(str(i)))
        except:
            #print("error")
            X = np.mat([float(x) for x in story_arc+[0]])
            plt.plot(X.T,'-k', label = 'story arc')
            n = len(story_arc)
            w = int(4 * np.floor(n/20) + 1)
            pass


    plt.title("$%s~=~{}$".format(str(Hurst)) % ("Hurst"))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$F(t)$')
    
    
def raw_smooth_not_plot(story_arc, Hurst):
    trends =[]
    y = integrate(story_arc)
    uneven = y.shape[1]%2
    if uneven:
        y = y[0,:-1]

    # afa
    #n = 500
    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)
    ## slope
    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    p = np.poly1d(np.polyfit(x, y, order))
    xp = np.linspace(0, len(x), len(x))

    #fig, ax = plt.subplots(2,1)

    #plt.figure(figsize=(12,20))
    #plt.subplot(311)
    X = np.mat([float(x) for x in story_arc])
    #plt.plot(X.T,'-k', label = 'story arc')
    n = len(story_arc)
    w = int(4 * np.floor(n/20) + 1)

    # format
    for i in range(2,5):
        try:
            _, trend_ww_1 = dm.detrending_method(X, w, i)
            trend = normalize(trend_ww_1).T
            trends.append(trend)
            #plt.plot(normalize(trend_ww_1).T, label = "$m = {}$".format(str(i)))
        except:
            #print("error")
            X = np.mat([float(x) for x in story_arc+[0]])
            #plt.plot(X.T,'-k', label = 'story arc')
            n = len(story_arc)
            w = int(4 * np.floor(n/20) + 1)
            pass

    return trends

def get_Hurst(story_arc):
    y = integrate(story_arc)
    uneven = y.shape[1]%2
    if uneven:
        y = y[0,:-1]

    # afa
    #n = 500
    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)
    ## slope
    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    #fig, ax = plt.subplots(2,1)

    X = np.mat([float(x) for x in story_arc])
    n = len(story_arc)
    w = int(4 * np.floor(n/20) + 1)

    # format
    for i in range(2,5):
        try:
            _, trend_ww_1 = dm.detrending_method(X, w, i)
        except:
            print("error")
            X = np.mat([float(x) for x in story_arc+[0]])
            n = len(story_arc)
            w = int(4 * np.floor(n/20) + 1)
            pass

        
    for i in range(2,5):
        _, trend_ww_1 = dm.detrending_method(X, w, i)
    
    H = round(np.polyfit(x, y, 1)[0],2)
    return H


def draw_figure(story_arc,sentimethod,workname):
    """
    
    draw plots for smoothed story arcs, sentiment value histogram and Hurst fit
    
    """
    y = integrate(story_arc)
    uneven = y.shape[1]%2
    if uneven:
        y = y[0,:-1]

    # afa
    #n = 500
    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)
    ## slope
    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    p = np.poly1d(np.polyfit(x, y, order))
    xp = np.linspace(0, len(x), len(x))


    plt.figure(figsize=(12,20))
    plt.subplot(311)
    X = np.mat([float(x) for x in story_arc])
    plt.plot(X.T,'-k', label = 'story arc', color="gray", alpha=0.7)
    n = len(story_arc)
    w = int(4 * np.floor(n/20) + 1)

    # format
    for i in range(2,5):
        try:
            _, trend_ww_1 = dm.detrending_method(X, w, i)
            plt.plot(normalize(trend_ww_1).T, label = "$m = {}$".format(str(i)))
        except:
            print("error")
            X = np.mat([float(x) for x in story_arc+[0]])
            plt.plot(X.T,'-k', label = 'story arc')
            n = len(story_arc)
            w = int(4 * np.floor(n/20) + 1)
            pass
    

    plt.title("$%s~Story~Arc~{}$".format(str(sentimethod)) % (workname))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$F(t)$')
    
    plt.subplot(323)
    M = np.mean(story_arc)
    SD =  np.std(story_arc)
    n, bins, _ = plt.hist(story_arc, optimal_bin(story_arc) ,density = True , facecolor = 'gray', edgecolor = 'w')
    
    Y = norm.pdf(bins, M, SD)
    plt.plot(bins, Y, 'k-', linewidth=1.5)
    plt.ylabel('$Sentiment~score$')
    plt.ylabel('$Density$')

    
    plt.subplot(324)
    plt.plot(xp, p(xp), 'k-', linewidth = 2,zorder = 0)
    plt.scatter(x, y, c = 'r', s = 50, zorder = 1)
    plt.title('$H = {}$'.format(round(np.polyfit(x, y, 1)[0],2)))
    plt.xlabel('$Log(w)$')
    plt.ylabel('$LogF(w)$')
    
    plt.tight_layout()



## other stuff
import unidecode
#

# Utils for sentiment arc extraction etc.

def clean_column(df,column_name, new_name):
    """
    removes special characters, accents and changes to lowercase for a df column
    
    df: a dataframe
    column_name: column you want to clean
    new_name: name of the new column
    """
    
    # split by $c: since from it follows author information
    df[new_name] = df[column_name].str.split('c:', expand = True)[0]
    
    # replace special characters but not spaces
    df[new_name]=df[new_name].str.replace(r'[^\w ]', '', regex=True)
    
    df[new_name]=df[new_name].str.replace(r'_', '', regex=True) 
    df[new_name]=df[new_name].apply(lambda x: unidecode.unidecode(x).lower() if type(x) != float else x)
    
    # multiple spaces into one
    df[new_name]=df[new_name].apply(lambda x: " ".join(x.split()) if type(x) != float else x)
    
    return df


# for HathiTrust ids

def id_encode(docid): 
    return docid.replace(":", "+").replace("/", "=").replace(",", ".")

def id_decode(docid): 
    return docid.replace("+", ":").replace("=", "/").replace(",", ".")



# create path 
def create_path(row, path):
    batch = str(row["batch"])
    if len(batch) == 1: # add zero if batch 0-9 ==> 00-09
        batch = "0" + batch
    filename = row["filename"]
    filepath = path+batch+"/"+filename+".txt"
    return filepath