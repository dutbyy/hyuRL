import concurrent.futures
import multiprocessing
import grpc
from proto import predictor_pb2
from proto import predictor_pb2_grpc
import asyncio


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
        request = predictor_pb2.InferenceReq()
        response = await self.stub.Inference(request)
        return response.err_msg
  
    async def inference(self):
        # print("infer")
        return await self.make_request()
            
        
                
async def tread_request():    
    import timeit
    caller = AsyncGRPCCaller()
    for i in range(5):
        try:
            a = timeit.default_timer()
            tasks = [asyncio.create_task(caller.inference()) for i in range(1)]
            b = timeit.default_timer()
            outs = await asyncio.gather(*tasks)
            c = timeit.default_timer()
            print(f"create task eplasde {(b-a)*1000}ms, gather eplasde {(c-b)*1000}ms, total {(c-a)*1000}ms")
        except Exception as e: 
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
    main()
    # start_threads(16)
