
import numpy as np
import torch, h5py, os
from collections import namedtuple

def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]

def pad_array(a, max_length, PAD=0):
    """
    a (array[int32])
    """
    return np.concatenate((a, [PAD]*(max_length - len(a))))

def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

#xs = np.array([np.r_[1], np.r_[1, 2, 3], np.r_[2, 1]])


class SlotData():
    def __init__(self, trips, times, ratios, S, distances, maxlen=200):
        """
        trips (n, *): each element is a sequence of road segments
        times (n, ): each element is a travel cost
        ratios (n, 2): end road segments ratio
        S (138, 148) or (num_channel, 138, 148): traffic state matrix
        """
        ## filter out the trips that are too long
        idx = [i for (i, trip) in enumerate(trips) if len(trip) <= maxlen]
        self.trips = trips[idx]
        self.times = times[idx]
        self.ratios = ratios[idx]
        self.distances = distances[idx]
        self.S = torch.tensor(S, dtype=torch.float32)
        ## (1, num_channel, height, width)
        if self.S.dim() == 2:
            self.S.unsqueeze_(0).unsqueeze_(0)
        elif self.S.dim() == 3:
            self.S.unsqueeze_(0)
        ## re-arrange the trips by the length in reverse order
        idx = argsort(trips)
        self.trips = self.trips[idx]
        self.times = torch.tensor(self.times[idx], dtype=torch.float32)
        self.ratios = torch.tensor(self.ratios[idx], dtype=torch.float32)
        self.distances = torch.tensor(self.distances[idx], dtype=torch.float32)

        self.ntrips = len(self.trips)
        self.start = 0


    def random_emit(self, batch_size):
        """
        Input:
          batch_size (int)
        ---
        Output:
          SD.trips (batch_size, seq_len)
          SD.times (batch_size,)
          SD.ratios (batch_size, seq_len)
          SD.S (num_channel, height, width)
        """
        SD = namedtuple('SD', ['trips', 'times', 'ratios', 'S', 'distances'])
        start = np.random.choice(max(1, self.ntrips-batch_size+1))
        end = min(start+batch_size, self.ntrips)

        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        distances = self.distances[start:end]
        ratios = torch.ones(trips.shape)
        ratios[:, 0] = self.ratios[start:end, 0]
        row_idx = list(range(trips.shape[0]))
        col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
        ratios[row_idx, col_idx] = self.ratios[start:end, 1]
        return SD(trips=trips, times=times, ratios=ratios, S=self.S, distances=distances)

    def order_emit(self, batch_size):
        """
        Reset the `start` every time the current slot has been traversed
        and return none.

        Input:
          batch_size (int)
        ---
        Output:
          SD.trips (batch_size, seq_len)
          SD.times (batch_size,)
          SD.ratios (batch_size, seq_len)
          SD.S (num_channel, height, width)
        """
        if self.start >= self.ntrips:
            self.start = 0
            return None
        SD = namedtuple('SD', ['trips', 'times', 'ratios', 'S', 'distances'])
        start = self.start
        end = min(start+batch_size, self.ntrips)
        self.start += batch_size

        trips = pad_arrays(self.trips[start:end])
        times = self.times[start:end]
        distances = self.distances[start:end]
        ratios = torch.ones(trips.shape)
        ratios[:, 0] = self.ratios[start:end, 0]
        row_idx = list(range(trips.shape[0]))
        col_idx = list(map(lambda t:len(t)-1, self.trips[start:end]))
        ratios[row_idx, col_idx] = self.ratios[start:end, 1]
        return SD(trips=trips, times=times, ratios=ratios, S=self.S, distances=distances)


class DataLoader():
    def __init__(self, trainpath, num_slot=71):
        """
        trainpath (string): The h5file path
        num_slot (int): The number of slots in a day
        """
        self.trainpath = trainpath
        self.num_slot = num_slot
        self.slotdata_pool = []
        ## `weights[i]` is proportional to the number of trips in `slotdata_pool[i]`
        self.weights = None
        ## The length of `slotdata_pool`
        self.length = 0
        ## The current index of the order emit
        self.order_idx = -1

    def read_file(self, fname):
        """
        Reading one h5file and appending the data into `slotdata_pool`. This function
        should only be called by `read_files()`.
        """
        with h5py.File(fname) as f:
            for slot in range(1, self.num_slot+1):
                S = np.rot90(f["/{}/S".format(slot)][...]).copy()
                n = f["/{}/ntrips".format(slot)][...]
                if n == 0: continue
                trips = [f["/{}/trip/{}".format(slot, i)][...] for i in range(1, n+1)]
                times = [f["/{}/time/{}".format(slot, i)][...] for i in range(1, n+1)]
                ratios = [f["/{}/ratio/{}".format(slot, i)][...] for i in range(1, n+1)]
                distances = [f["/{}/distance/{}".format(slot, i)][...] for i in range(1, n+1)]
                self.slotdata_pool.append(
                    SlotData(np.array(trips), np.array(times), np.array(ratios), S,
                             np.array(distances)))

    def read_files(self, fname_lst):
        """
        Reading a list of h5file and appending the data into `slotdata_pool`.
        """
        for fname in fname_lst:
            fname = os.path.basename(fname)
            print("Reading {}...".format(fname))
            self.read_file(os.path.join(self.trainpath, fname))
            print("Done.")
        self.weights = np.array(list(map(lambda s:s.ntrips, self.slotdata_pool)))
        self.weights = self.weights / np.sum(self.weights)
        self.length = len(self.weights)
        self.order = np.random.permutation(self.length)
        self.order_idx = 0

    def random_emit(self, batch_size):
        """
        Return a batch of data randomly.
        """
        i = np.random.choice(self.length, p=self.weights)
        return self.slotdata_pool[i].random_emit(batch_size)

    def order_emit(self, batch_size):
        """
        Visiting the `slotdata_pool` according to `order` and returning the data in the
        slot `slotdata_pool[i]` orderly.
        """
        i = self.order[self.order_idx]
        data = self.slotdata_pool[i].order_emit(batch_size)
        if data is None: ## move to the next slot
            self.order_idx += 1
            if self.order_idx >= self.length:
                self.order_idx = 0
                self.order = np.random.permutation(self.length)
            i = self.order[self.order_idx]
            data = self.slotdata_pool[i].order_emit(batch_size)
        return data
