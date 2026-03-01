import numpy as np

class StateLogger:
    def __init__(self, keys):
        """
        Example:
            logger = StateLogger(["s", "P", "K", "z", "z_hat", "e"])
        """
        self.data = {key: [] for key in keys}


    def append(self, *args, **kwargs):
        """
        Supports two formats:
        - append(s=s_k1_k1, P=P_k1_k1)
        - append([['s', s_k1_k1], ['P', P_k1_k1]])
        """
        # if a list of pairs is given as the first positional argument
        if args:
            pairs = args[0]
            for key, value in pairs:
                if key in self.data and value is not None:
                    self.data[key].append(value)

        # plus the standard keyword-arguments
        for key, value in kwargs.items():
            if key in self.data and value is not None:
                self.data[key].append(value)

    def get_stat(self, keys):
        if isinstance(keys, str):
            keys = [keys]
            single = True
        else:
            single = False
    
        results = []
        for key in keys:
            lst = self.data.get(key, None)
            if not lst:
                results.append(None)
                continue
    
            arr = np.array(lst)
            results.append(arr)
    
        if single:
            return results[0]
        return tuple(results)
