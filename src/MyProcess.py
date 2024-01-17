from __future__ import print_function
import multiprocessing as mp
import traceback

class MyProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            #self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            #self._cconn.send((e, tb))
            print(e, tb)
            # raise e  # You can still rise this exception if you need to

    def start(self) -> None:
        try:
            mp.Process.start(self)
            #self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            #self._cconn.send((e, tb))
            print(e, tb)

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception