import grpc
from proto import predictor_pb2
from proto import predictor_pb2_grpc

class PredictorServiceServicer(predictor_pb2_grpc.PredictorServiceServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('inits over')
        
    def Inference(self, request, context):
        self.times += 1   
        t = self.times
        import time
        time.sleep(1)
        print(f"inference {t}")
        rsp = predictor_pb2.InferenceRsp(err_msg=f'success: {t}')
        return rsp

def serve():
    from concurrent import futures
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=256))
    service = PredictorServiceServicer()
    predictor_pb2_grpc.add_PredictorServiceServicer_to_server(service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
