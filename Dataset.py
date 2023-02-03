import numpy as np
import tensorflow as tf
import os
import time
from Signal import Signal

class Dataset:
    
    def __init__(self, path, sf, order, bp_lo, bp_hi, notch, window, stride, freq, bw, harms, apply_snr, lab_rels):
        self.lab_rels = lab_rels
        self.data = []
        if type(path) == str:
            fname = os.path.basename(path)
            name, sess, file_freq, date = fname.replace('.txt', '').split(' ')
            if freq is None:
                freq = float(file_freq.replace('Hz', ''))
            each_signal = Signal(
                name.strip(),
                int(sess.strip()),
                time.strptime(date.strip(), '%Y-%m-%d')
            )
            each_signal.load_raw(path, self.lab_rels)
            each_signal.process(sf, order, bp_lo, bp_hi, notch)
            each_signal.make_fvs(window, stride, freq, bw, harms, apply_snr)
            self.data.append(each_signal)
        else:
            for p in path:
                fname = os.path.basename(p)
                name, sess, file_freq, date = fname.replace('.txt', '').split(' ')
                if freq is None:
                    freq = float(file_freq.replace('Hz', ''))
                each_signal = Signal(
                    name.strip(),
                    int(sess.strip()),
                    time.strptime(date.strip().replace('-', ' '), '%Y %m %d')
                )
                each_signal.load_raw(p, self.lab_rels)
                each_signal.process(sf, order, bp_lo, bp_hi, notch)
                each_signal.make_fvs(sf, window, stride, freq, bw, harms, apply_snr)
                self.data.append(each_signal)
    
    def get_metadata(self):
        return [{"name": d.get_subject(), "session": d.get_session(), "date": d.get_date_string()} for d in self.data]
    
    def list_metadata(self):
        print('subject\t-\tsession\t-\tdate')
        for m in self.get_metadata():
            print(f'{m["name"]}\t-\t{m["session"]}\t-\t{m["date"]}')
            
    def get_metadata_strat(self, subjects=None, sessions=None, labels=None):
        #for stratification with train_test_split, returns a vector with strings "subjectsession" 
        #repeated as many times as feature vectors that user and session combination contain
        if subjects is not None and type(subjects) != list:
            raise ValueError('subjects should be list or None')
        if sessions is not None and type(sessions) != list:
            raise ValueError('sessions should be list or None')
        if labels is not None and type(labels) != list:
            raise ValueError('labels should be list or None')
            
        if labels is not None:
            labels = self.get_transform_labels(labels)
        
        return np.array([
            f'{d.get_subject()}{d.get_session()}' for d in self.data for _ in range(d.get_Y(labels).shape[0]) if (
                (True if subjects is None else d.subject in subjects) and
                (True if sessions is None else d.session in sessions)
            )
        ])
    
    def get_fv(self, source, channels=None, subjects=None, sessions=None, labels=None):
        if channels is not None and type(channels) != list:
            raise ValueError('channels should be list or None')
        if subjects is not None and type(subjects) != list:
            raise ValueError('subjects should be list or None')
        if sessions is not None and type(sessions) != list:
            raise ValueError('sessions should be list or None')
        if labels is not None and type(labels) != list:
            raise ValueError('labels should be list or None')
            
        if labels is not None:
            labels = self.get_transform_labels(labels)
            
        if source ==  'time':
            return self.get_fv_time(channels, subjects, sessions, labels)
        elif source == 'freq':
            return self.get_fv_freq(channels, subjects, sessions, labels)
        else:
            raise ValueError(f'Invallid source {source}, should be one of time or freq.')
            
    def get_fv_time(self, channels, subjects, sessions, labels):
        return np.vstack([
            d.get_time_X_stack(channels, labels) for d in self.data if (
                (True if subjects is None else d.subject in subjects) and
                (True if sessions is None else d.session in sessions)
            )
        ])
    
    def get_fv_freq(self, channels, subjects, sessions, labels):
        return np.vstack([
            d.get_freq_X_stack(channels, labels) for d in self.data if (
                (True if subjects is None else d.subject in subjects) and
                (True if sessions is None else d.session in sessions)
            )
        ])
    
    def get_onehot(self, subjects=None, sessions=None, labels=None):
        if subjects is not None and type(subjects) != list:
            raise ValueError('subjects should be list or None')
        if sessions is not None and type(sessions) != list:
            raise ValueError('sessions should be list or None')
        if labels is not None and type(labels) != list:
            raise ValueError('labels should be list or None')
            
        if labels is not None:
            labels = self.get_transform_labels(labels)
        return np.vstack([
            d.get_Y(labels) for d in self.data if (
                (True if subjects is None else d.subject in subjects) and
                (True if sessions is None else d.session in sessions)
            )
        ])
    
    def get_transform_labels(self, origin):
        if type(origin) == list:
            return [self.lab_rels[o] for o in origin]
        else:
            return self.lab_rels[origin]
