import pickle
import grpc
import time
import asyncio
import timeit
from networkx import dag_longest_path
import torch
import numpy as np
from typing import Dict, Union
from proto import predictor_pb2
from proto import predictor_pb2_grpc
from src.tools.gpu import auto_move

def common_serialize(data: Dict[str, np.ndarray]) -> bytes:
    np_list = predictor_pb2.NumpyList()
    def serialize_item(item: Dict, pre_name=None):
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
    def __init__(self, host, port, aio=True):
        if aio:
            self.channel = grpc.aio.insecure_channel(f'{host}:{port}')
        else:
            self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = predictor_pb2_grpc.PredictorServiceStub(self.channel)

    async def predict(self, inputs):
        request = predictor_pb2.InferenceReq(data=common_serialize(inputs))
        inference_response = await self.stub.Inference(request)
        return common_deserialize(inference_response.data), inference_response.err_code

    async def log_probs(self, data, action, mask):
        inputs = {
            'logits': data,
            'action': action,
            'mask': mask
        }
        request = predictor_pb2.InferenceReq(data=common_serialize(inputs))
        inference_response = await self.stub.LogProbs(request)
        return common_deserialize(inference_response.data)

    def update_weight(self, weights):
        def tfunc(stub, weights):
            return stub.UpdateWeight(predictor_pb2.UpdateWeightReq(weight=pickle.dumps(weights)))
        return tfunc(self.stub, weights)

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
        elif isinstance(value, np.ndarray):
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
        self.batch_size = 8
        self.start_time = None
        self.timeout = 10
        self.times = 0
        super().__init__(*args, **kwargs)

    async def Inference(self, request, context):
        self.times += 1
        a = timeit.default_timer()
        future = asyncio.Future()
        await self._data_queue.put([common_deserialize(request.data), future])
        data = await future
        b = timeit.default_timer()
        rsp_data = common_serialize(data)
        rsp = predictor_pb2.InferenceRsp(err_code=int((b-a)*1000), err_msg=f'latency : {(b-a)* 1000}')
        rsp.data.CopyFrom(rsp_data)
        return rsp

    # def Inference1(self, request, context):
    #     self.times += 1
    #     t = self.times
    #     data = self._model(auto_move(common_deserialize(request.data), 'cuda'))
    #     rsp = predictor_pb2.InferenceRsp(data=common_serialize(data), err_msg=f'success: {t}')
    #     return rsp

    # async def LogProbs(self, request, context):
    #     t = self.times
    #     inputs = common_deserialize(request.data)
    #     data = self._model._network.log_probs(*inputs.values())
    #     rsp = predictor_pb2.InferenceRsp(data=common_serialize(data), err_msg=f'success: {t}')
    #     return rsp

    async def UpdateWeight(self, request, context):
        # self._model.update_weight(request.model_name, request.weights)
        state_dict = pickle.loads(request.weight)
        print(f"state dict is {state_dict.keys()}")
        self._model._network.load_state_dict(state_dict)
        response = predictor_pb2.UpdateWeightRsp(
            weight = pickle.dumps(self._model._network.state_dict()),
            err_code=0,
            err_msg="Weights updated"
        )
        print("Updated weights finished")
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
                            tmp_timeout = 0.9 * (self.timeout/1000 - diff) if len(requests) else 10
                            def prints():
                                # print('await for')
                                pass
                            prints()
                            request = await asyncio.wait_for(self._data_queue.get(), timeout=tmp_timeout)
                            # request = self._data_queue.get_nowait()
                            requests.append(request)
                            if len(requests) == 1:
                                self.start_time = time.time()
                        except Exception as e:
                            # await asyncio.sleep(0.005)
                            pass
                    elif len(requests) > 0:
                        break
                '''
                now = time.time() * 1000
                qsize = self._data_queue.qsize()
                # print(f'qsize is {qsize}')
                if qsize >= self.batch_size:
                    requests = [self._data_queue.get_nowait() for i in range(qsize)]
                    # print('manzu le jixu')
                elif qsize > 0:
                    if not self.start_time:
                        self.start_time = now
                        await asyncio.sleep(0.0001)
                        continue
                    elif now - self.start_time * 1000 > self.timeout:
                        # print('超时了 jixu')
                        requests = [self._data_queue.get_nowait() for i in range(qsize)]
                    else :
                        await asyncio.sleep(0.0001)
                        continue
                else:
                    await asyncio.sleep(0.0001)
                    continue
                '''
                # print(requests)


                # print(f'check batch! {len(requests)} eplased time is {(time.time() - self.start_time) * 1000}ms')

                def batch_inference(requests):
                    inputs = convert_to_batch_state([it[0] for it in requests])
                    # auto_move(inputs, 'cuda')
                    results, _ = self._model.inference(inputs)
                    results = split_outputs(results)
                    # results = [{} for i in requests]
                    for idx, (_, future) in enumerate(requests):
                        result = results[idx]
                        if not future.cancelled() and not future.done():
                            future.set_result(result)
                batch_inference(requests)
                self.start_time = time.time()
                requests = []
            except Exception as e :
                print(e)
                raise e

async def serve(model):
    from concurrent import futures
    server = grpc.aio.server()
    service = PredictorServiceServicer(model)
    predictor_pb2_grpc.add_PredictorServiceServicer_to_server(service, server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    batch_inference = asyncio.create_task(service.start_batch_inference())
    def callback(future):
        print("batch inference exception!!!")
        exit(-1)
    batch_inference.add_done_callback(callback)
    await server.wait_for_termination()

def main():
    from local.net import get_model
    print("prepate to server")
    asyncio.run(serve(get_model()))

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()','restats')
