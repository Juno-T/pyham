import threading

class AlternateLock():
    def __init__(self, initial_thread):
        self.allow = initial_thread
        self.cond = threading.Condition()
        
    def acquire_for(self, thread):
        self.cond.acquire()
        while self.allow!=thread:
            self.cond.wait()
    
    def release_to(self, thread):
        self.allow=thread
        self.cond.notifyAll()
        self.cond.release()
