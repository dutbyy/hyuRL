import concurrent.futures
from typing import Dict
import numpy as np
import torch
import multiprocessing
from fsspec import Callback
import grpc
from sympy import fu, true
from proto import predictor_pb2
from proto import predictor_pb2_grpc
import asyncio

def common_serialize(data: Dict[str, np.ndarray]) -> bytes:
    np_list = predictor_pb2.NumpyList()
    def serialize_item(item, pre_name=None):
        for k, v in item.items():
            current_name = f"{pre_name}.{k}" if pre_name is not None else k
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach().numpy()
            if isinstance(v, dict):
                serialize_item(v, pre_name=f'{current_name}')
            elif v is None :
                continue
            else:
                np_item = predictor_pb2.NumpyData(
                    name = current_name,
                    dtype = str(v.dtype),
                    array_data = np.array(v).tobytes(),
                    shape = v.shape,
                )
                np_list.np_arrays.append(np_item)
    serialize_item(data)
    return np_list

class AsyncGRPCCaller:
    """A class for making asynchronous gRPC calls."""
    def __init__(self, host='localhost', port=50051):
        self.host = host
        self.port = port
        self.create_channel_and_stub()
        
    def handle_asyncio_exception(self, loop, context):
        # 处理来自事件循环的异常
        exception = context.get('exception')
        if exception:
            print(f"Caught exception: {exception}")
        else:
            print("Caught non-exception")
        # 可以在这里添加更多的错误处理逻辑
    
    def create_channel_and_stub(self):
        self.channel = grpc.aio.insecure_channel(f'{self.host}:{self.port}')
        self.stub = predictor_pb2_grpc.PredictorServiceStub(self.channel)

    async def make_request(self):
        request = predictor_pb2.InferenceReq(data=common_serialize(
            {"feature_a": torch.rand(4)}
        ))
        response = await self.stub.Inference(request)
        return response.err_msg
    
    async def inference(self):
        # print("infer")
        return await self.make_request()
            
        
                
async def tread_request():    
    import timeit
    caller = AsyncGRPCCaller()
    for i in range(100):
        try:
            a = timeit.default_timer()
            tasks = [asyncio.create_task(caller.inference()) for i in range(1000)]
            b = timeit.default_timer()
            outs = await asyncio.gather(*tasks)
            c = timeit.default_timer()
            print(f"create task eplasde {(b-a)*1000}ms, gather eplasde {(c-b)*1000}ms, total {(c-a)*1000}ms")
            # print(outs, flush=true)
        except Exception as e: 
            # print(e)
            raise e
        
def main():
    asyncio.run(tread_request())

def start_threads(num_threads):
    import multiprocessing
    threads = []
    for _ in range(num_threads):
        
        thread = multiprocessing.Process(target=main)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

# # 使用示例
if __name__ == '__main__':
    import time
        # time.sleep(.1)
    try:
        main()
    except:
        pass
    # start_threads(1)
