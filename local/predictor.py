import pickle
from typing import Dict, Union
import grpc
from torch import Tensor
import torch
from proto import predictor_pb2
from proto import predictor_pb2_grpc
import time
import asyncio
import timeit
from asyncio import shield
import numpy as np
from src.tools.gpu import auto_move

def common_serialize(data: Dict[str, np.ndarray]) -> bytes:
    np_list = predictor_pb2.NumpyList()
    def serialize_item(item, pre_name=None):
        for k, v in item.items():
            current_name = f"{pre_name}.{k}" if pre_name is not None else k
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach().numpy() 
            if isinstance(v, dict):
                serialize_item(v, pre_name=f'{current_name}')
            else :
                np_item = predictor_pb2.NumpyData(
                    name = current_name, 
                    dtype = str(v.dtype),
                    array_data = np.array(v).tobytes(), 
                    shape = v.shape,
                )
                np_list.np_arrays.append(np_item)
    serialize_item(data)
    return np_list


def common_deserialize(np_list: predictor_pb2.NumpyList) -> Dict[str, Union[np.ndarray, Dict]]:
    data_dict = {}

    def deserialize_item(item: predictor_pb2.NumpyData, current_dict: Dict):
        dtype = np.dtype(item.dtype)
        array = np.frombuffer(item.array_data, dtype=dtype).reshape(item.shape)

        name = item.name
        if '.' in name:
            parts = name.split('.')
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            last_part = parts[-1]
            current_dict[last_part] = array
        else:
            current_dict[name] = array

    for item in np_list.np_arrays:
        deserialize_item(item, data_dict)
    
    return data_dict



class PredictorClient:
    def __init__(self, host, port):
        self.channel = grpc.aio.insecure_channel(f'{host}:{port}')
        self.stub = predictor_pb2_grpc.PredictorServiceStub(self.channel)
        
    async def predict(self, inputs):
        # nplist = common_serialize(inputs)
        # print(f'predicting: {inputs}')
        request = predictor_pb2.InferenceReq()
        request.data.CopyFrom(common_serialize(inputs))
        # print('!!!!!!', request)
        inference_response = await self.stub.Inference(request)
        # print("Inference response:", inference_response.data)
        return common_deserialize(inference_response.data), inference_response.err_code
    
    async def log_probs(self, data, action, mask):
        request = predictor_pb2.InferenceReq(data=pickle.dumps([data, action, mask]))
        inference_response = await self.stub.LogProbs(request)
        # print("Inference response:", inference_response.data)
        return pickle.loads(inference_response.data)
    
    def update_weight(self, weights):
        request = predictor_pb2.UpdateWeightReq(weight=pickle.dumps(weights))
        inference_response = self.stub.UpdateWeight(request)
        # print("Inference response:", inference_response.data)
        return 

def convert_to_batch_state(states):
    assert len(states) > 0
    batch_state_dict = {}
    for key, state in states[0].items():
        if isinstance(state, dict):
            sub_states = [s[key] for s in states]
            batch_state_dict[key] = convert_to_batch_state(sub_states)
        else:
            batch_state_dict[key] = np.stack([s[key] for s in states])
    return batch_state_dict
            
def split_outputs(results):
    outputs = []
    for key, value in results.items():
        if isinstance(value, dict):
            inner_outputs = split_outputs(value)
            for i, inner_output in enumerate(inner_outputs):
                if i < len(outputs):
                    outputs[i][key] = inner_output
                else:
                    outputs.append({key: inner_output})
        elif isinstance(value, Tensor):
            rows = np.split(value, len(value))
            for i, row in enumerate(rows):
                row = np.squeeze(row)
                if i < len(outputs):
                    outputs[i][key] = row
                else:
                    outputs.append({key: row})
        elif value is None:
            continue
        else:
            raise ValueError(f"Unsupported output type: {type(value)}")
    return outputs

class PredictorServiceServicer(predictor_pb2_grpc.PredictorServiceServicer):
    def __init__(self, model, *args, **kwargs):
        self._data_queue = asyncio.Queue()
        self._model = model
        self.batch_size = 32
        self.start_time = None
        self.timeout = 25
        self.times = 0
        print('inits')
        super().__init__(*args, **kwargs)
        print('inits over')
        
    async def Inference(self, request, context):
        self.times += 1   
        t = self.times
        # data = ''
        a = timeit.default_timer()
        future = asyncio.Future()
        await self._data_queue.put([common_deserialize(request.data), future])
        data = await future
        b = timeit.default_timer()
        rsp_data = common_serialize(data)
        rsp = predictor_pb2.InferenceRsp(err_code=int((b-a)*1000), err_msg=f'latency : {(b-a)* 1000}')
        rsp.data.CopyFrom(rsp_data)
        return rsp
    
    # async def Inference(self, request, context):
    #     self.times += 1   
    #     t = self.times
    #     data = self._model(pickle.loads(request.data))
    #     rsp = predictor_pb2.InferenceRsp(data=pickle.dumps(data), err_msg=f'success: {t}')
    #     return rsp
    
    async def LogProbs(self, request, context):
        t = self.times
        inputs = pickle.loads(request.data) 
        output = self._model.log_probs(*inputs)
        rsp = predictor_pb2.InferenceRsp(data=pickle.dumps(output), err_msg=f'success: {t}')
        return rsp

    def UpdateWeight(self, request, context):
        # self._model.update_weight(request.model_name, request.weights)
        self._model.load_state_dict(pickle.loads(request.weight))
        response = predictor_pb2.UpdateWeightRsp(
            err_code=0,
            err_msg="Weights updated"
        )
        return response

    async def start_batch_inference(self):
        requests = []
        while True:
            try : 
                while len(requests) < self.batch_size:
                    if len(requests) == 0 or not self.start_time:
                        self.start_time = time.time()
                    diff = time.time() - self.start_time 
                    if diff * 1000 < self.timeout:
                        try:
                            a = time.perf_counter()
                            tmp_timeout = (self.timeout/1000 - diff) if len(requests) else 10
                            # tmp_timeout = 0.01 if len(requests) else 1
                            request = await asyncio.wait_for(self._data_queue.get(), timeout=tmp_timeout) 
                            requests.append(request)
                            # print('appends ok')
                        except Exception as e:
                        # except asyncio.exceptions.TimeoutError as e:
                            # print(f'error is {e}')
                            # b = time.perf_counter()
                            # print(f"[{e}] wait for eplased time is {1000*(b-a):.2} ms, set timeout is {1000*tmp_timeout}ms")
                            # print(f'starttime: {self.start_time}, wating timeout : {len(requests)} diff is {diff *1000} sleep for {1000 * (self.timeout/1000 - diff) if len(requests) else 10 * 1000}')
                            pass
                    else:
                        break
                # print(f'check batch! {len(requests)} eplased time is {diff * 1000}ms')
                
                
                atime = timeit.default_timer() * 1000
                inputs = convert_to_batch_state([it[0] for it in requests])
                auto_move(inputs, 'cuda')
                btime = timeit.default_timer() * 1000
                results = self._model(inputs)     
                ctime = timeit.default_timer() * 1000
                results = split_outputs(results)
                dtime = timeit.default_timer() * 1000
                print(f"{time.time() * 1000 % 60000} batch {len(requests)} infer time: {round(btime - atime, 2)} {round(ctime - btime, 2)} {round(dtime - ctime, 2)}")
                
                for idx, (request, future) in enumerate(requests):
                    result = results[idx]
                    if not future.cancelled() and not future.done():
                        future.set_result(result)
                # print('set result over!!!!!')
                self.start_time = time.time()
                requests = []
            except Exception as e :
                print(e)
                raise e

async def serve(model):
    server = grpc.aio.server()
    service = PredictorServiceServicer(model)
    predictor_pb2_grpc.add_PredictorServiceServicer_to_server(service, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    batch_inference = asyncio.create_task(service.start_batch_inference())
    # batch_inference = asyncio.create_task(service.start_batch_logprobs())
    def callback(feture):
        print("batch inference exception!!!")
        exit(-1) 
    batch_inference.add_done_callback(callback)
    await server.wait_for_termination()

if __name__ == '__main__':
    # import tracemalloc
    # tracemalloc.start()
    print("prepate to server")
    asyncio.run(serve())
