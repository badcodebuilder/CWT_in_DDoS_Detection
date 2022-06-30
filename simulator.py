import datetime
import random
from copy import deepcopy
from threading import Thread
from time import sleep
from typing import List, Tuple

from tqdm import trange

t_IP = Tuple[int, int, int, int]

class Simulator:
    def __init__(self, interval: float) -> None:
        self.req_queue: List[Tuple[int, int]]   = []
        self.data: List[Tuple[int, float]]      = []
        self.labels: List[bool]                 = []
        self.is_attack: int                     = 0
        self.interval: float                    = interval

    def __randSrcIP(self) -> int:
        return self.__ip2int(tuple([random.randint(0,255) for _ in range(4)]))

    def __ip2int(self, ip: t_IP) -> int:
        return (ip[0]<<24 + ip[1]<<16 + ip[2]<<8 + ip[3])

    def normalVisit(
        self,
        src_ip: int,
        dst_ip: int) -> None:
        count = random.randint(4,32)
        for _ in range(count):
            self.req_queue.append((src_ip, dst_ip))
            sleep(random.random()*20*self.interval)

    def abnormalVisit(
        self,
        dst_ip: int) -> None:
        count = random.randint(1000000, 10000000)
        for _ in range(count):
            self.is_attack = 1
            # Append是符合实际的，因为每次来的请求不可能缓存下来，而且这里都是虚拟的IP地址
            self.req_queue.append((self.__randSrcIP(), dst_ip))
            # sleep(1e-4*self.interval)
        self.is_attack = 0

    def visitSim(self, dst_ip: int):
        while True:
            sleep(0.1*self.interval)
            t = random.random()
            if t < 2e-4:
                attack = Thread(target=self.abnormalVisit, args=[dst_ip])
                attack.start()
            elif t < 1e-2:
                visit = Thread(target=self.normalVisit, args=[self.__randSrcIP(), dst_ip])
                visit.start()
            else:
                pass

    def record(self):
        with open("data/record_{}.csv".format(datetime.datetime.now().strftime("%H_%M")), 'w', encoding="utf-8") as f:
            f.write("src_ip_count,dst_ip_count,label\n")
            f.flush()
            for _ in trange(3600):
                sleep(self.interval)
                prev_req_queue = deepcopy(self.req_queue)
                self.req_queue.clear()

                pkt_count = len(prev_req_queue)
                src_ip_count = len(set([req[0] for req in prev_req_queue]))
                dst_ip_count = len(set([req[1] for req in prev_req_queue]))

                f.write("{},{},{}\n".format(src_ip_count, (dst_ip_count/pkt_count) if pkt_count > 0 else 0, self.is_attack))
                f.flush()

    def run(self) -> int:
        """
            该模型确实也很像实际的攻击以及采样的模型

            模拟模块会不停的有人来访问，采样模型每隔一段时间采样，采样完成就结束整个进程
        """
        record_thread = Thread(target=self.record)
        visitor_thread = Thread(target=self.visitSim, args=[self.__randSrcIP()])
        visitor_thread.daemon = True

        record_thread.start()
        visitor_thread.start()
        record_thread.join()
        return 0

if __name__ == "__main__":
    s = Simulator(1e-1)
    s.run()
