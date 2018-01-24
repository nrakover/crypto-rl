'''
API for working with agent state-space.
'''

class State:
    '''
    Abstraction for a state object.
    '''

    STATE_WINDOW_IN_SEC = 0 #604800    # 1 week

    def __init__(self, snapshot, allocation):
        self.snapshot = snapshot
        self.allocation = allocation

    def __str__(self):
        return "{0}::{1}".format(self.snapshot.get_timestamp(), self.allocation)

    @staticmethod
    def create(history, timestep, allocation):
        return State(history[timestep], allocation)    #TODO: implement
    
    @staticmethod
    def find_starting_timestep(history):
        '''
        Finds the earliest starting timestep in a history such that the state window can be computed.
        '''
        # TODO: binary search instead, based on history granularity or finest possible granularity (60 sec)
        first_timestamp = history[0].get_timestamp()
        timestamp = first_timestamp
        t = 0
        while timestamp - first_timestamp < State.STATE_WINDOW_IN_SEC:
            t += 1
            timestamp = history[t].get_timestamp()
        return t
