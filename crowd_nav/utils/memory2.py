import random


class ReplayBuffer:
    """
    将不同人数的experience分别放入不同的list中，以保证批量训练
    """

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = {}
        self.size = 0
        self.device = device

    def push(self, item):
        key = str(item[0].shape[0])
        if self.size < self.capacity:
            self.add_to_memory(key, item)
        else:
            # delete an old experience
            deleted = False
            while not deleted:
                rand_key = random.choice(list(self.memory.keys()))
                if len(self.memory[rand_key]) > 100:
                    self.memory[rand_key].pop(0)
                    # add new item
                    self.add_to_memory(key, item)
                    deleted = True

    def add_to_memory(self, key, item):
        self.size += 1
        if key not in self.memory.keys():
            self.memory[key] = [item]
        else:
            self.memory[key].append(item)

    def get_chunks(self, arr, chunk_size):
        length = len(arr)
        res = []
        for i in range(0, len(arr), chunk_size):
            if i + chunk_size < length:
                res.append(arr[i : i + chunk_size])
            else:
                res.append(arr[i:])
        return res

    def sample_batches(self, batch_size, num_batches=100, sample_all=False):
        """
        :batch_num: number of batch to sample
        :sample_all: True if want to sample all batches
        """
        inputs = []
        values = []
        for key, experiences in self.memory.items():
            li = [i for i in range(len(experiences))]
            random.shuffle(li)
            chunks = self.get_chunks(li, batch_size)
            for chunk in chunks:
                inputs.append([experiences[index][0] for index in chunk])
                values.append([experiences[index][1] for index in chunk])
        if not sample_all:
            choices = random.choices(
                range(len(inputs)),
                k=num_batches if num_batches <= len(inputs) else len(inputs),
            )
            sub_inputs = [inputs[c] for c in choices]
            sub_values = [values[c] for c in choices]
            return sub_inputs, sub_values
        return inputs, values

    def clear(self):
        self.memory = {}

    def __len__(self):
        return self.size
