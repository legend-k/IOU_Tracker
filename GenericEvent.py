import inspect
# remove after debug
import pdb

class GenericEvent(object):

    # initialization function
    def __init__(self, **signature):
        '''
        Event_ as of now only accepts keyworded arguments
        Does not accept positional arguments for now: TODO
        '''
        self._signature = signature
        self._argnames = set(signature.keys())
        self._handlers = []

    # destructor function
    def __del__(self):
        self._argnames = None
        self._handlers = None
        self._signature = None
        self._num_handlers = None

    # returns the signature of the event in string format
    def _kwargs_str(self):
        return ", ".join(str(k) + "=" + str(v) for k, v in self._signature.items())

    # overloading the increment operator--> +=
    def __iadd__(self, handler):
        params = inspect.signature(handler).parameters
        valid = True
        argnames = set(n for n in params.keys())
        if argnames != self._argnames:
            valid = False
        for p in params.values():
            if p.kind == p.VAR_KEYWORD:
                valid = True
                break
            if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                valid = False
                break
        if not valid:
            raise ValueError("Listener must have these arguments: (%s)"
                             % self._kwargs_str())
        self._handlers.append(handler)
        self._num_handlers = len(self._handlers)
        return self

    # overloading the decrement operator--> -=
    def __isub__(self, handler):
        self._handlers.remove(handler)
        self._num_handlers = len(self._handlers)
        return self

    # overloading the function call operator--> ()
    def __call__(self, *args, **kwargs):
        if args or set(kwargs.keys()) != self._argnames:
            #pdb.set_trace() # BREAKPOINT
            raise ValueError("This EventHook must be called with these " +
                             "keyword arguments: (%s)" % self._kwargs_str())
        for handler in self._handlers[:]:
            handler(**kwargs)

        return None

    # returns the event signature
    def __repr__(self):
        return "EventHook(%s)" % self._kwargs_str()

    # property that retrieves if any handlers have subscribed to the event
    @property
    def isSubscribed(self):
        if self._num_handlers > 0:
            return True
        else:
            return False

if __name__ == "__main__":
    pass
