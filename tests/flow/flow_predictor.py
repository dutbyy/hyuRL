import pickle
import gzip
import grpc
import time
import asyncio
import timeit
import torch
import numpy as np
from typing import Dict, Union
from tests.proto import predictor_pb2
from tests.proto import predictor_pb2_grpc


def common_serialize(data: Dict[str, np.ndarray]) -> bytes:
    np_list = predictor_pb2.NumpyList()

    def serialize_item(item: Dict, pre_name=None):
        for k, v in item.items():
            current_name = f"{pre_name}.{k}" if pre_name is not None else k
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach().numpy()
            if isinstance(v, dict):
                serialize_item(v, pre_name=f"{current_name}")
            elif v is None:
                continue
            else:
                np_item = predictor_pb2.NumpyData(
                    name=current_name,
                    dtype=str(v.dtype),
                    array_data=np.array(v).tobytes(),
                    shape=v.shape,
                )
                np_list.np_arrays.append(np_item)

    serialize_item(data)
    return np_list


def common_deserialize(
    np_list: predictor_pb2.NumpyList,
) -> Dict[str, Union[np.ndarray, Dict]]:
    data_dict = {}

    def deserialize_item(item: predictor_pb2.NumpyData, current_dict: Dict):
        dtype = np.dtype(item.dtype)
        array = np.frombuffer(item.array_data, dtype=dtype).reshape(item.shape)
        name = item.name
        if "." in name:
            parts = name.split(".")
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


def pretocuda(nested_structure, cuda=True):
    import tree

    # print(nested_structure)
    if torch.cuda.is_available():
        return tree.map_structure(
            lambda x: torch.as_tensor(x).cuda(), nested_structure
        )
    return tree.map_structure(lambda x: torch.as_tensor(x), nested_structure)


def convert_to_batch_state(states):
    assert len(states) > 0
    batch_state_dict = {}
    for key, state in states[0].items():
        if isinstance(state, dict):
            sub_states = [s[key] for s in states]
            batch_state_dict[key] = convert_to_batch_state(sub_states)
        else:
            try:
                batch_state_dict[key] = np.stack([s[key] for s in states])
            except Exception as e:
                print(f"exception key is {key}")
                raise e
    return pretocuda(batch_state_dict, False)


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
        elif isinstance(value, torch.Tensor):
            rows = np.split(value.cpu().numpy(), len(value))
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


class LocalPredictorClient:
    def __init__(self, name2model):
        self.name2model = name2model

    async def predict(self, request, context):
        pass


class PredictorClient:
    def __init__(self, host, port, aio=True):
        if aio:
            self.channel = grpc.aio.insecure_channel(
                f"{host}:{port}",
                options=[
                    (
                        "grpc.max_send_message_length",
                        -1,
                    ),  # 发送的最大消息长度，-1 表示无限制
                    ("grpc.max_receive_message_length", -1),
                ],
            )
        else:
            self.channel = grpc.insecure_channel(
                f"{host}:{port}",
                options=[
                    (
                        "grpc.max_send_message_length",
                        -1,
                    ),  # 发送的最大消息长度，-1 表示无限制
                    ("grpc.max_receive_message_length", -1),
                ],
            )
        self.stub = predictor_pb2_grpc.PredictorServiceStub(self.channel)

    async def predict(self, state_dict):
        request = predictor_pb2.InferenceReq(
            model_name=state_dict["model"], data=common_serialize(state_dict["obs"])
        )
        inference_response = await self.stub.Inference(request)
        return common_deserialize(inference_response.data), inference_response.err_code

    def update_weight(self, model_name, weights, msg=""):
        pickle_weight = gzip.compress(pickle.dumps(weights))
        req = predictor_pb2.UpdateWeightReq(
            model_name=model_name, weight=pickle_weight, extra_msg=msg
        )
        return self.stub.UpdateWeight(req)


class PredictorServiceServicer(predictor_pb2_grpc.PredictorServiceServicer):
    def __init__(self, name2model, *args, **kwargs):
        self._data_queue = {name: asyncio.Queue() for name in name2model}
        self._name2model = name2model
        self.batch_size = 4
        self.start_time = None
        self.timeout = 10
        self.times = 0
        super().__init__(*args, **kwargs)

    async def Inference(self, request, context):
        self.times += 1
        a = timeit.default_timer()
        future = asyncio.Future()
        await self._data_queue[request.model_name].put(
            [common_deserialize(request.data), future]
        )
        data = await future
        b = timeit.default_timer()
        rsp_data = common_serialize(data)
        rsp = predictor_pb2.InferenceRsp(err_code=0, err_msg=f"latency : {(b-a)* 1000}")
        rsp.data.CopyFrom(rsp_data)
        return rsp

    async def UpdateWeight(self, request, context):
        model_name = request.model_name
        print(f"Updating weights for model: [{model_name}], {request.extra_msg}")
        weights = pickle.loads(gzip.decompress(request.weight))
        flow_model = self._name2model.get(model_name)

        with torch.no_grad():
            for target_p, p in zip(flow_model._model._network.parameters(), weights):
                target_p.copy_(torch.from_numpy(p))
        response = predictor_pb2.UpdateWeightRsp(
            weight=b"", err_code=0, err_msg=f"Updated {model_name}'s weight."
        )
        print("Updated weights finished")
        return response

    async def start_batch_inference(self, model_name):
        requests = []
        start_time = None
        while True:
            await asyncio.sleep(0.001)
            while len(requests) < self.batch_size:
                if len(requests) == 0 or not start_time:
                    start_time = time.time()
                diff = time.time() - start_time
                if diff * 1000 < self.timeout:
                    try:
                        tmp_timeout = (
                            0.9 * (self.timeout / 1000 - diff) if len(requests) else 1
                        )
                        request = await asyncio.wait_for(
                            self._data_queue[model_name].get(), timeout=tmp_timeout
                        )
                        requests.append(request)
                        if len(requests) == 1:
                            start_time = time.time()
                    except Exception as e:
                        pass
                elif len(requests) > 0:
                    break

            def batch_inference(requests):
                inputs = convert_to_batch_state([it[0] for it in requests])
                a = time.time() * 1e6
                results = self._name2model[model_name].predict(inputs)
                latency = time.time() * 1e6 - a
                results = split_outputs(results)
                for idx, (_, future) in enumerate(requests):
                    result = results[idx]
                    if not future.cancelled() and not future.done():
                        future.set_result(result)

            batch_inference(requests)
            start_time = time.time()
            requests = []


async def serve(name2model):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),  # 发送的最大消息长度，-1 表示无限制
            ("grpc.max_receive_message_length", -1),
        ]
    )
    for name, model in name2model.items():
        model.setstate_predict(model.__getstate__())
    service = PredictorServiceServicer(name2model)
    predictor_pb2_grpc.add_PredictorServiceServicer_to_server(service, server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    for model_name in name2model.keys():
        batch_inference = asyncio.create_task(service.start_batch_inference(model_name))

        def callback(future):
            print("batch inference exception!!!")
            exit(-1)

        batch_inference.add_done_callback(callback)
    await server.wait_for_termination()


def main(flow_config, builder):
    predict_model_names = []
    for actor_name, actor_config in flow_config["actor_config"].items():
        for model_learn_config in actor_config["training_models"]:
            predict_model_names.append(model_learn_config["model_name"])
        if actor_config["inference_models"]:
            predict_model_names.extend(actor_config["inference_models"])
    name2model = {}
    for model_name in predict_model_names:
        flow_model = flow_config["algorithm"]["flow_model"](model_name, builder)
        name2model[model_name] = flow_model
    print("Predictor server starting.")
    asyncio.run(serve(name2model))


if __name__ == "__main__":
    from hyurl.tools.common import fix_print

    fix_print()
    from tests.flow.env_config import flow_config

    main(flow_config, flow_config["builder"])
