class Task:
    def __init__(self, id, deadline, penalty) -> None:
        self.id = id
        self.deadline = deadline
        self.penalty = penalty


def f(A,t):
    # A is a set of deadlines of tasks
    cnt = 0
    for a in A:
        cnt = cnt+1 if a.deadline < t else cnt
    return cnt

def scheduler(tasks):
    sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
    
    A = []
    B = []
    for t,d in enumerate(sorted_tasks):
        A.append(d)
        if f(A,t) < t:
            continue
        else:
            A.remove(d)
            B.append(d)
    return A,B
    

if __name__=="__main__":
    tasks = [Task(1,4,70),Task(2,2,60),Task(3,4,50),Task(4,3,40),
             Task(5,1,30),Task(6,4,20),Task(7,6,10)]
    
    schedule,rejected = scheduler(tasks)
    print(f"optimal schedule: {[t.id for t in schedule]}")
    
    print(f"total penalty: {sum([t.penalty for t in rejected])}")
    