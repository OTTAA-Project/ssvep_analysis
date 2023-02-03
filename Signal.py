import numpy as np
import scipy.signal as sgn
import tensorflow as tf
import os
import time

def reshape_with_stride(arr, window, stride):
    return np.vstack([
        arr[np.newaxis, i:i+window] for i in range(0, arr.shape[0], stride) if i+window <= arr.shape[0]
    ])

def pick_freq(fbins, fft, f, bw, harmonics = [], apply_snr = False, get_bins = False):
    mask = []
    if 1 not in harmonics:
        harmonics.append(1) #include fundamental, 0th harmonic of f
    for fbin in fbins:
        mask.append(any([fmatch - bw <= fbin <= fmatch + bw for fmatch in [f*h for h in harmonics]]))
    mask = np.array(mask) #to be able to use BITWISE OPS
    if get_bins:
        return fbins[mask]
    if apply_snr:
        return fft[:, mask] / fft[:, ~mask].mean(axis=-1)[:, np.newaxis]
    else:
        return fft[:, mask]

class Signal:
    def __init__(self, subject, session, date):
        self.subject = subject
        self.session = session
        self.date = date
        
    def load_raw(self, path, lab_rels = {}):
        chans = []
        labs = []
        with open(path, encoding = 'utf-8') as f:
            text = f.read()
            lines = text.split('\n')
            for l in lines:
                if len(l) == 0:
                    continue
                if l[0] != '%' and len(l.split(',')) == 11: #first skipped lines start with % skipped lines
                    chans.append([float(n) for n in l.split(',')[1:5]]) #channels are 1-4
                    each_label = int(l.split(',')[8]) #labels are 8
                    #change labels to range(0, num_labels)
                    try:
                        each_label_trans = lab_rels[each_label]
                    except KeyError:
                        lab_rels[each_label] = len(lab_rels.keys())
                        each_label_trans = lab_rels[each_label]
                    labs.append(each_label_trans) 
        
        self.sparse_labels_rels = lab_rels
        self.sparse_labels_rels_back = {v: k for k, v in lab_rels.items()}
        sparse_labels = np.array(labs)
        
        stim_pos = np.where(sparse_labels != self.get_transform_label(99))[0]
        start_stim_pos = stim_pos.min()
        end_stim_pos = stim_pos.max()
        
        self.raw = np.array(chans)[start_stim_pos:end_stim_pos].T
        self.sparse_labels = sparse_labels[start_stim_pos:end_stim_pos]
        
    def process(self, sf, order, bp_lo, bp_hi, notch):
        nq = sf/2
        b_bp, a_bp = sgn.iirfilter(order, [bp_lo/nq, bp_hi/nq], btype='band', ftype='butter')
        b_nt, a_nt = sgn.iirnotch(notch/nq, Q=notch, fs = sf) #Q=w0/bw, setting it to the same value of notch makes a 1Hz BW stopband

        self.processed = sgn.filtfilt(b_bp, a_bp, sgn.filtfilt(b_nt, a_nt, self.raw, axis=1), axis=1)
    
    def make_fvs(self, sf, window, stride, freq, bw, harms, apply_snr):
        self.make_fv_time(window, stride)
        self.make_fv_freq(sf, freq, bw, harms, apply_snr)
        
    def make_fv_time(self, window, stride):
        if self.processed is None:
            raise ValueError('No processed data to turn into FVs')
        self.time_X = {
            'ch1': reshape_with_stride(self.processed[0], window, stride),
            'ch2': reshape_with_stride(self.processed[1], window, stride),
            'ch3': reshape_with_stride(self.processed[2], window, stride),
            'ch4': reshape_with_stride(self.processed[3], window, stride)
        }
        lab_reshape = reshape_with_stride(self.sparse_labels, window, stride)
        self.categ_labels = np.zeros( (lab_reshape.shape[0], len(self.sparse_labels_rels.keys())) )
        for lvi, lab_vect in enumerate(lab_reshape):
            vals, counts = np.unique(lab_vect, return_counts = True)
            rels_counts = counts/counts.sum()
            for v, c in zip(vals, rels_counts):
                self.categ_labels[lvi, v] = c
        
        assert self.time_X['ch1'].shape[0] == \
            self.time_X['ch2'].shape[0] == \
            self.time_X['ch3'].shape[0] == \
            self.time_X['ch4'].shape[0] == \
            self.categ_labels.shape[0], f'''Unmatching shapes:
            X:ch1: {self.time_X['ch1'].shape}
            X:ch2: {self.time_X['ch2'].shape}
            X:ch3: {self.time_X['ch3'].shape}
            X:ch4: {self.time_X['ch4'].shape}
            y: {self.categ_labels.shape}
            '''
    
    def make_fv_freq(self, sf, freq, bw, harms, apply_snr):
        if self.time_X is None:
            raise ValueError('No time FV data to turn into freq FV')
        fbins = np.fft.rfftfreq(self.time_X['ch1'].shape[1], d=1/sf) 
        self.freq_X = {}
        for ch in self.get_chans():
            if freq is None:
                self.freq_X[ch] = abs(np.fft.rfft(self.time_X[ch]))
            else:
                self.freq_X[ch] = pick_freq(fbins, abs(np.fft.rfft(self.time_X[ch])), freq, bw, harms, apply_snr)
        self.freq_bins = fbins
        if freq is None:
            self.freq_bins_X = fbins
        else:
            self.freq_X_bins = pick_freq(fbins, None, freq, bw, harms, apply_snr, get_bins = True)
    
    def get_subject(self):
        return self.subject
    
    def get_session(self):
        return self.session
    
    def get_date(self):
        return self.date
    
    def get_date_string(self):
        return time.strftime('%Y-%m-%d', self.date)
    
    def get_sparse_labels(self):
        return self.sparse_labels
    
    def get_origin_label(self, transformed):
        try:
            return self.sparse_labels_rels_back[transformed]
        except KeyError:
            raise KeyError(f'There is no transformed label {transformed}')
    
    def get_transform_label(self, origin):
        try:
            return self.sparse_labels_rels[origin]
        except KeyError:
            raise KeyError(f'There is no original label {origin}')
            
    def get_raw(self):
        return self.raw
    
    def get_processed(self):
        return self.processed
    
    def get_chans(self):
        return list(self.get_time_X().keys())
    
    def get_time_X(self, channel=None, labels=None):
        if channel is None:
            return self.time_X
        else:
            try:
                channel_data = self.time_X[channel]
            except KeyError:
                raise KeyError(f'The channel requested {channel} does not exist')
            
            if labels is None:
                return channel_data
            elif type(labels) == list:
                return channel_data[np.isin(self.get_Y().argmax(axis=1), labels)]
            else:
                return channel_data[np.where(self.get_Y().argmax(axis=1) == labels)[0]]
    
    def get_time_X_stack(self, channels=None, labels=None):
        if channels is None:
            stacked = np.hstack(list(self.get_time_X().values()))
        elif type(channels) == list:
            stacked = np.hstack([v for k, v in self.get_time_X.items() if k in channels])
        else:
            raise ValueError('channels parameter should be a list or None')
            
        if labels is None:
            return stacked
        elif type(labels) == list:
            return stacked[np.isin(self.get_Y().argmax(axis=1), labels)]
        else:
            return stacked[np.where(self.get_Y().argmax(axis=1) == labels)[0]]
    
    def get_freq_X(self, channel=None, labels=None):
        if channel is None:
            return self.freq_X
        else:
            try:
                channel_data = self.freq_X[channel]
            except KeyError:
                raise KeyError(f'The channel requested {channel} does not exist')
                
            if labels is None:
                return channel_data
            elif type(labels) == list:
                return channel_data[np.isin(self.get_Y().argmax(axis=1), labels)]
            else:
                return channel_data[np.where(self.get_Y().argmax(axis=1) == labels)[0]]
    
    def get_freq_X_stack(self, channels=None, labels=None):
        if channels is None:
            stacked = np.hstack(list(self.get_freq_X().values()))
        elif type(channels) == list:
            stacked = np.hstack([v for k, v in self.get_freq_X().items() if k in channels])
        else:
            raise ValueError('channels parameter should be a list or None')
        
        if labels is None:
            return stacked
        elif type(labels) == list:
            return stacked[np.isin(self.get_Y().argmax(axis=1), labels)]
        else:
            return stacked[np.where(self.get_Y().argmax(axis=1) == labels)[0]]
    
    def get_freq_bins(self):
        return self.freq_bins
    
    def get_freq_X_bins(self):
        return self.freq_X_bins

    def get_Y(self, labels=None):
        if labels is None:
            return self.categ_labels
        elif type(labels) == list:
            return self.categ_labels[np.isin(self.categ_labels.argmax(axis=1), labels)]
        else:
            return self.categ_labels[np.where(self.categ_labels.argmax(axis=1) == labels)[0]]
            