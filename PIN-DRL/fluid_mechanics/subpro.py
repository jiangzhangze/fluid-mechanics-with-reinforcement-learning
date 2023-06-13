import subprocess
from multiprocessing import Process, Queue

def run_subprocess(cmd, queue):
    """运行子进程并将结果放入队列中"""
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    queue.put(result)

if __name__ == '__main__':
    cmd1 = "ls /"
    cmd2 = "cat /etc/passwd"
    queue = Queue()

    # 在两个进程中分别运行两个命令
    p1 = Process(target=run_subprocess, args=(cmd1, queue))
    p2 = Process(target=run_subprocess, args=(cmd2, queue))
    p1.start()
    p2.start()

    # 获取两个进程返回的结果
    result1 = queue.get()
    result2 = queue.get()

    # 打印结果
    print("Command 1 output:\n" + result1.stdout.decode('utf-8'))
    print("Command 2 output:\n" + result2.stdout.decode('utf-8'))