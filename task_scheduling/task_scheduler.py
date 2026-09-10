from collections import defaultdict
import heapq
import logging

log = logging.getLogger("task_scheduler")


class TaskScheduler:
    def __init__(self):
        self._heap = []
        self._seq = 0
        self._deadline = {}
        self._remaining = {}
        self._dependents = defaultdict(list)
        self._consumed = set()

    def _push_eligible(self, task_id):
        heapq.heappush(self._heap, (self._deadline[task_id], self._seq, task_id))
        self._seq += 1
        log.info("eligible task_id=%s deadline=%s", task_id, self._deadline[task_id])

    def add_tasks(self, tasks):
        log.info("add start count=%s", len(tasks))
        for task_id, deadline, deps in tasks:
            self._deadline[task_id] = deadline
            unmet = 0
            for dep in deps:
                if dep not in self._consumed:
                    unmet += 1
                    self._dependents[dep].append(task_id)
            self._remaining[task_id] = unmet
            log.info("add task_id=%s deadline=%s unmet=%s", task_id, deadline, unmet)
            if unmet == 0:
                self._push_eligible(task_id)
        log.info("add done eligible=%s", len(self._heap))

    def update_deadline(self, task_id, new_deadline):
        if task_id not in self._deadline or task_id in self._consumed:
            log.info("update skip task_id=%s", task_id)
            return False
        self._deadline[task_id] = new_deadline
        log.info("update task_id=%s deadline=%s unmet=%s", task_id, new_deadline, self._remaining[task_id])
      #  if self._remaining[task_id] == 0:
      #      self._push_eligible(task_id)
        return True

    def consume_task(self):
        while self._heap:
            deadline, _seq, task_id = heapq.heappop(self._heap)
            if task_id in self._consumed or deadline != self._deadline[task_id]:
                log.info("consume skip stale task_id=%s deadline=%s", task_id, deadline)
                continue
            self._consumed.add(task_id)
            for waiter in self._dependents[task_id]:
                self._remaining[waiter] -= 1
                if self._remaining[waiter] == 0:
                    self._push_eligible(waiter)
            log.info("consume task_id=%s deadline=%s eligible=%s", task_id, deadline, len(self._heap))
            return task_id
        log.info("consume none available")
        return -1
from collections import defaultdict
import heapq
import logging

log = logging.getLogger("task_scheduler")


class TaskScheduler:
    def __init__(self):
        self._heap = []
        self._seq = 0
        self._deadline = {}
        self._remaining = {}
        self._dependents = defaultdict(list)
        self._consumed = set()

    def _push_eligible(self, task_id):
        heapq.heappush(self._heap, (self._deadline[task_id], self._seq, task_id))
        self._seq += 1
        log.info("eligible task_id=%s deadline=%s", task_id, self._deadline[task_id])

    def add_tasks(self, tasks):
        log.info("add start count=%s", len(tasks))
        for task_id, deadline, deps in tasks:
            self._deadline[task_id] = deadline
            unmet = 0
            for dep in deps:
                if dep not in self._consumed:
                    unmet += 1
                    self._dependents[dep].append(task_id)
            self._remaining[task_id] = unmet
            log.info("add task_id=%s deadline=%s unmet=%s", task_id, deadline, unmet)
            if unmet == 0:
                self._push_eligible(task_id)
        log.info("add done eligible=%s", len(self._heap))

    def update_deadline(self, task_id, new_deadline):
        if task_id not in self._deadline or task_id in self._consumed:
            log.info("update skip task_id=%s", task_id)
            return False
        self._deadline[task_id] = new_deadline
        log.info("update task_id=%s deadline=%s unmet=%s", task_id, new_deadline, self._remaining[task_id])
      #  if self._remaining[task_id] == 0:
      #      self._push_eligible(task_id)
        return True

    def consume_task(self):
        while self._heap:
            deadline, _seq, task_id = heapq.heappop(self._heap)
            if task_id in self._consumed or deadline != self._deadline[task_id]:
                log.info("consume skip stale task_id=%s deadline=%s", task_id, deadline)
                continue
            self._consumed.add(task_id)
            for waiter in self._dependents[task_id]:
                self._remaining[waiter] -= 1
                if self._remaining[waiter] == 0:
                    self._push_eligible(waiter)
            log.info("consume task_id=%s deadline=%s eligible=%s", task_id, deadline, len(self._heap))
            return task_id
        log.info("consume none available")
        return -1
from collections import defaultdict
import heapq
import logging

log = logging.getLogger("task_scheduler")


class TaskScheduler:
    def __init__(self):
        self._heap = []
        self._seq = 0
        self._deadline = {}
        self._remaining = {}
        self._dependents = defaultdict(list)
        self._consumed = set()

    def _push_eligible(self, task_id):
        heapq.heappush(self._heap, (self._deadline[task_id], self._seq, task_id))
        self._seq += 1
        log.info("eligible task_id=%s deadline=%s", task_id, self._deadline[task_id])

    def add_tasks(self, tasks):
        log.info("add start count=%s", len(tasks))
        for task_id, deadline, deps in tasks:
            self._deadline[task_id] = deadline
            unmet = 0
            for dep in deps:
                if dep not in self._consumed:
                    unmet += 1
                    self._dependents[dep].append(task_id)
            self._remaining[task_id] = unmet
            log.info("add task_id=%s deadline=%s unmet=%s", task_id, deadline, unmet)
            if unmet == 0:
                self._push_eligible(task_id)
        log.info("add done eligible=%s", len(self._heap))

    def update_deadline(self, task_id, new_deadline):
        if task_id not in self._deadline or task_id in self._consumed:
            log.info("update skip task_id=%s", task_id)
            return False
        self._deadline[task_id] = new_deadline
        log.info("update task_id=%s deadline=%s unmet=%s", task_id, new_deadline, self._remaining[task_id])
      #  if self._remaining[task_id] == 0:
      #      self._push_eligible(task_id)
        return True

    def consume_task(self):
        while self._heap:
            deadline, _seq, task_id = heapq.heappop(self._heap)
            if task_id in self._consumed or deadline != self._deadline[task_id]:
                log.info("consume skip stale task_id=%s deadline=%s", task_id, deadline)
                continue
            self._consumed.add(task_id)
            for waiter in self._dependents[task_id]:
                self._remaining[waiter] -= 1
                if self._remaining[waiter] == 0:
                    self._push_eligible(waiter)
            log.info("consume task_id=%s deadline=%s eligible=%s", task_id, deadline, len(self._heap))
            return task_id
        log.info("consume none available")
        return -1

