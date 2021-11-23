
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from read_hpf import Hpf


def bins_hpf(stroke, other, name, n_bins, samp_rate, start_t=0, abs=False):
    """
    stroke is displacenemt- vector of measured displacements. it is same length as other.
    other is the measured variable.
    takes hpf object and returns
    returns a dict of data in separate bins, and a list of what the bins are
    """

    hist, bins = np.histogram(stroke, bins=n_bins)
    stroke_inds = find_max_min(stroke)
    bins = np.round(bins, 5)

    if len(stroke_inds) ==0:
        print('len stroke inds=0')
        return None
    #pd cut gives a list of bin indices
    bin_inds = pd.cut(stroke, bins=bins,  right=False, include_lowest=True,labels=False)
    bin_inds = np.nan_to_num(bin_inds, posinf=n_bins-1)
    bin_inds = bin_inds.astype('int')

    t=start_t
    t_lst = []
    max_other = np.max(other)
    min_other = np.min(other)
    bin_dict={}

    bin_dict[name+' - right']=[]
    bin_dict[name+' - left']=[]

    for i, (mx, mn, nxt) in enumerate(stroke_inds):
        left = other[mx:mn] #get down data for that attribute
        right = other[mn:nxt] #get up data for that attribute
        t = start_t+ mx/samp_rate
        t_lst.append(t)
        if abs:
            left = np.abs(left)
            right = np.abs(right)
        # get data for each stroke, left
        l_data_bins = map_to_bin(bin_inds[mx:mn], left, n_bins) #list of lists
        l_avg_bins = get_avg_data_bins(l_data_bins) #list of avgs for each value of displacemnt
        # get data for right stroke
        r_data_bins = map_to_bin(bin_inds[mn:nxt], right, n_bins)
        r_avg_bins = get_avg_data_bins(r_data_bins)
        #append to bin_dict.
        bin_dict[name+' - right'].append(r_avg_bins)
        bin_dict[name+' - left'].append(l_avg_bins)

    return bin_dict, bins, t_lst, stroke_inds



def convert_disp(array, arr_max, arr_min):
    """
    conversts displacement in V to mm
    """
    mx = max(array) #positive
    mn = min(array) # negative
    new_span = arr_max - arr_min
    new_middle = (arr_max + arr_min)/2
    middle = (mx + mn)/2
    span = mx-mn
    array = (new_span)*(array - middle)/span + new_middle
    return array


def get_avg_data_bins(bins):
    """
    takes in bins
    outputs average for each bin
    """
    avgs=[]
    for i in range(len(bins)):
        bin = bins[i]
        if len(bin)>0:
            avg = np.mean(bin)
        else:
            avg = np.nan
        avgs.append(avg)

    return np.array(avgs)

def scaled(arr):
    """
    scaled so arr max = 1 and arr_min = 0
    """
    arr = arr-np.min(arr) # all numbers now positive or zero
    arr = arr/np.max(arr) #gives largest magnitude number 1
    return arr

def find_max_min(arr):
    """
    this function finds a set of max min and next for each 'wave' in what
    might not be monotonically increasing or decreasing data.
    it depends on the stroke being normalised to between 0 and 1
    so the diff (0.1) is the right size
    """
    stroke = None
    max_start=None
    min_start=None
    max_end=None
    min_end=None
    stroke_inds=[]
    arr = scaled(arr) # now between 0 and 1
    diff = 0.1

    if arr[0] > 1 - diff:
        print('starting near top')
        max_next = False
        max_end = 1
    elif arr[0] < 0+diff:
        print('starting near bottom')
        max_next = True
    else:
        print('starting in middle')
        max_next=True

    for i, x in enumerate(arr):
        #find max
        if x > 1-diff and max_next and not max_start:
            # just got near top.
            max_start = i
            max_next = False

        elif x < 1-diff  and max_start and i > max_start+10:
            #starting to go down.
            max_end = i
            mx = (max_end + max_start)//2
            if stroke:
                stroke.append(mx)
                stroke_inds.append(stroke)
            stroke = [mx+1]
            max_start = None

        elif x< 0+diff and max_end:
            #getting near bottom of stroke
            min_start = i
            max_end=None

        elif x>0+diff and min_start and i > min_start+10:
            #starting to go back up
            min_end = i
            mn = (min_end+min_start)//2
            if stroke:
                stroke.append(mn)
            min_start = None
            max_next = True

        else:
            continue

    return stroke_inds

def get_expt_info(exp_no, path, spreadsheet_name='Running conditions.xlsx'):
    """
    reads pings file on running conditions to provide dictionary to look up
    running conditions. load for friciton calculation.
    """
    if spreadsheet_name[-4:]=='xlsx':
        # requires install of 'openpxl' library via pip or conda
        df = pd.read_excel(path+spreadsheet_name)
    else:  #easier with csv or text files.
        df = pd.read_csv(path+spreadsheet_name)
    # converts only columns that are all number to numeric values
    df.apply(pd.to_numeric, errors='ignore')
    df = df.drop(['Date'], axis=1)
    df.set_index('file name', inplace=True)
    exp_info = dict(df.loc[exp_no])
    exp_info['Load']
    default_vals={'Load':10, 'Friction Adjustment (N/V)':10, 'Charge amp adjustment (pC/V)':1}
    for key in default_vals.keys():
        try: # replacement doesn't work with excel file
            var = exp_info[key] # does it have a value?
            exp_info[key] = float(exp_info[key]) # is it a number or a list?
            print(exp_no, key, var)
        except KeyError: # no value
            exp_info[key] = default_vals[key]
            print(f"can't find {key} for expt. {exp_no}")
        except ValueError: #is list of 3 for load
            loads = var.split('/')
            load = float(loads[int(len(loads)/2)])
            exp_info[key]=load
            print(f"load is list {loads} for expt. {exp_no}, chosing {load}")
    return exp_info


def map_to_bin(bin_inds, data, n_bins):
    """
    takes bin indices, puts relevant data in relevant bin
    bins is 1+max value in bin_inds lists of lists
    """
    assert len(bin_inds) == len(data), "data (len {}) and bin indices array (len {}) should be the same length".format(len(data), len(bin_inds))
    bins = [[] for i in range(n_bins)] #make bins for up and down data
    for i in range(len(data)):
        bins[bin_inds[i]].append(data[i])
    return bins



def get_bin_dicts(hpf, exp_no, exp_info):
    """
    get binned data from hpf file
    put it in to a list of lists, with a key that has the direction and type of data

    """

    print('in get bin dicts')
    o_chan=1
    s_chan = 0
    r_max = 12.5

    data_dict={} # to store data in bins for both left and right strokes for each type of data.
    bins_dict={} # to store what the bins are for each type of data.
    samp_rate = int(hpf.chan_info[o_chan]['PerChannelSampleRate']) #in Hz
    n_chan = hpf.num_channels - 1 # num data channels
    print(f'n_chan={n_chan}')

    load = exp_info['Load']
    ch_adj = exp_info['Charge amp adjustment (pC/V)']
    fr_adj = exp_info['Friction Adjustment (N/V)']
    data_ref = {0: 'Stroke', 1:'ES', 2:'Friction'}
    for chan in range(n_chan):
        name = data_ref[chan+1]
        stroke = hpf.data[:,s_chan]
        other = hpf.data[:,chan+1]

        if name == 'ES':
            s_dist = 6.8
            other = other*ch_adj
        elif name =='Friction':
            s_dist = 0
            other = other* (fr_adj/load)

        arr_max = r_max+s_dist
        arr_min = -r_max+s_dist
        stroke = convert_disp(stroke, arr_max, arr_min)

        samp_rate = int(hpf.chan_info[o_chan]['PerChannelSampleRate']) #in Hz
        n_bins = samp_rate//10 #
        stroke = convert_disp(stroke, arr_max, arr_min)
        bin_dict, bins, t_lst, stroke_inds = bins_hpf(stroke, other, name, n_bins, samp_rate)
        print('bins len = ', len(bins))
        for key in bin_dict.keys():
            # print(key, 'data', bin_dict[key])
            data_dict[key]=bin_dict[key]
            bins_dict[key]=bins

    return data_dict, bins_dict, t_lst

def make_data_df_from_bins_dict(data_dict, bins_dict, t_lst):
    """
    takes bins dict, key gives list of lists, each list is for a stroke,
    each stroke is a list of numbers corresponding to that bin
    bins_lst gives lower end of bin range,
    t_lst gives time since expt started in s
    returns dict where key is data type and direction, eg ES - right
    value is df with index t_list, columns as bins
    """
    df_dict={}
    for key in data_dict.keys():
        print('in make_data_df_from_bins_dict ', key)
        data_lsts = data_dict[key]
        bins=bins_dict[key]
        #build df
        df = pd.DataFrame(data_dict[key], columns=bins[:-1], index=t_lst)
        if np.isnan(df.values).sum()==df.size: # if is all nans, empty df
            print(f'key {key} is empty')
            continue
        df = clean_df(df) # replaces nans from empty bins with neighbouring val
        df_dict[key] = df

    return df_dict


def clean_df(df):
    """
    removes nans, and ensures all are the same length
    """
    n,p = df.shape
    row_nans = df.isnull().any(axis=1) # gives boolean mask by row
    df_nans = df.isnull() # gives full matrix of booleans true for nan

    for sample in range(n): # gives every sample iloc
        if row_nans.iloc[sample]: # there is a nan in this sample
            for ind in range(p):
                if df_nans.iloc[sample].iloc[ind]:
                    row = df.iloc[sample]
                    if sum(df_nans.iloc[sample])==p: #if row empty
                        return df

                    row = find_nearest_neighbour(row, df_nans.iloc[sample], ind)
                    df.iloc[sample]=row # put in the row with less nans
    return df




def find_nearest_neighbour(row, row_nans, ind):
    """
    takes in a row that has nans, and the index one of those nans is at.
    returns row with nan at ind replaced by nearest neighbour that is not a nan
    """
    diff = 1
    while True:
        if ind - diff >=0:
            lind = ind-diff
            if not row_nans.iloc[lind]: #ie if the next one down is not a nan
                row.iloc[ind] = row.iloc[lind]
                return row
        if ind + diff <len(row):
            rind=ind+diff
            if not row_nans.iloc[rind]: #ie if the next one up is not a nan
                row.iloc[ind]=row.iloc[rind]
                return row
        if ind -diff<0 and ind + diff > len(row):
            print('run out of row')
            print(row_nans)
            print(row)
            print(ind)
            print(sum(row))
            print(sum(row_nans))
            raise ValueError
            # return 'Fail'

        diff+=1


def read_bins_from_file(exp_no, bin_path):
    """
    Reads data from files in bin_path folder.
    For each file:
        Puts it in to data frames, one for each direction and channel.
    Usually: ES - right, ES - left, Friction - right, Friction - left
    four per hpf file.
    if it finds no binned files, returns FileNotFoundError
    so should try to read hpf file directly.
    """
    print('in read bin dicts from file')
    data_dfs={}
    found=False
    for file in os.listdir(bin_path):
        if 'bins' in file and exp_no in file:
            found=True
            file_tup = file.split('_')
            typ = file_tup[-2]
            df = pd.read_csv(bin_path+file, sep=',', index_col='stroke_n')

            for dirn in ['left', 'right']:
                df_d = df[df['dir']==dirn]
                df_d.index=df_d['time']
                df_d = df_d.drop(['dir', 'time'], axis=1)
                data_dfs[f'{typ} - {dirn}'] = df_d

    if not found:
        raise FileNotFoundError
    return data_dfs

def write_bins_to_file(exp_no, path, data_dfs):
    """
    has data frame with binned data.
    one data frame for each direction and each channel.
    writes each channel to a file with both directions.
    """
    typ_dict = {}
    for name in data_dfs.keys():
        typ, _, dir = name.split(' ') # gets data type and direction from dict key
        if not typ in typ_dict:
            typ_dict[typ]=[]
        # get data frame, size n strokes by p bins
        df = data_dfs[name]
        n, n_bins = df.shape
        print(f'writing {typ}, {dir}, {n} strokes, {n_bins} bins to file')
        df['dir']=[dir]*n # make direction column
        df['time']=df.index # make time column
        df.index = np.arange(n) # stroke numbers.
        typ_dict[typ].append(df)
    for typ in typ_dict.keys():
        df_d = pd.concat(typ_dict[typ])
        print(df_d.shape)
        fname = '{}_{}_bins.txt'.format(exp_no, typ)
        df_d.to_csv(path+fname, index_label='stroke_n')




def get_binned_data_from_path(exp_no, hpf_path, bin_path, exp_info):
    """
    should take exp_no and path and return binned data.
    should check for existence of pre binned data first.
    should generate and write bins if not present.
    exp_no - string with name of expt
    path - path to hpf files where expt hpf is
    """
    print(exp_no)
    try:
        data_dfs= read_bins_from_file(exp_no, bin_path)
        print(f'read {exp_no} from file {bin_path}')

    except FileNotFoundError:
        print('FileNotFoundError')
        file = exp_no+'.hpf'
        hpf = Hpf.init_from_hpf(hpf_path+file)
        data_dict, bins_dict, t_lst = get_bin_dicts(hpf, exp_no, exp_info)
        data_dfs = make_data_df_from_bins_dict(data_dict, bins_dict, t_lst)
        write_bins_to_file(exp_no, bin_path, data_dfs)
        data_dfs= read_bins_from_file(exp_no, bin_path)
    return data_dfs




if __name__ == '__main__':
    # exp_info = get_expt_info('0616_bk_hard_80_1000-2')
    # print(exp_info)
    from read_hpf import Hpf
    for day in ['0630', '0707', '0723', '0804', '0805', '0806', '0807', '0811', '0812', '0817', '0818', '0901','0914', '0915', '0916', '1104']:
    # for day in ['1104']:
        path = os.path.expanduser('~') +'/Data/ping/Tool steel/Test {}/'.format(day)
        if 'data' in os.listdir(path): # some have the hpf files in a sub folder data
            path = path+'data/'
        bin_path = os.path.expanduser('~') +'/Data/ping/Tool steel/Test {}/bins/'.format(day)
