# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/predictor.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15proto/predictor.proto\"8\n\tNumpyList\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1d\n\tnp_arrays\x18\x02 \x03(\x0b\x32\n.NumpyData\"K\n\tNumpyData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05\x64type\x18\x02 \x01(\t\x12\x12\n\narray_data\x18\x03 \x01(\x0c\x12\r\n\x05shape\x18\x04 \x03(\x05\"(\n\x0cInferenceReq\x12\x18\n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\n.NumpyList\"K\n\x0cInferenceRsp\x12\x10\n\x08\x65rr_code\x18\x01 \x01(\x05\x12\x0f\n\x07\x65rr_msg\x18\x02 \x01(\t\x12\x18\n\x04\x64\x61ta\x18\x03 \x01(\x0b\x32\n.NumpyList\"!\n\x0fUpdateWeightReq\x12\x0e\n\x06weight\x18\x01 \x01(\x0c\"D\n\x0fUpdateWeightRsp\x12\x0e\n\x06weight\x18\x01 \x01(\x0c\x12\x10\n\x08\x65rr_code\x18\x02 \x01(\x05\x12\x0f\n\x07\x65rr_msg\x18\x03 \x01(\t2\x9b\x01\n\x10PredictorService\x12)\n\tInference\x12\r.InferenceReq\x1a\r.InferenceRsp\x12(\n\x08LogProbs\x12\r.InferenceReq\x1a\r.InferenceRsp\x12\x32\n\x0cUpdateWeight\x12\x10.UpdateWeightReq\x1a\x10.UpdateWeightRspb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.predictor_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_NUMPYLIST']._serialized_start=25
  _globals['_NUMPYLIST']._serialized_end=81
  _globals['_NUMPYDATA']._serialized_start=83
  _globals['_NUMPYDATA']._serialized_end=158
  _globals['_INFERENCEREQ']._serialized_start=160
  _globals['_INFERENCEREQ']._serialized_end=200
  _globals['_INFERENCERSP']._serialized_start=202
  _globals['_INFERENCERSP']._serialized_end=277
  _globals['_UPDATEWEIGHTREQ']._serialized_start=279
  _globals['_UPDATEWEIGHTREQ']._serialized_end=312
  _globals['_UPDATEWEIGHTRSP']._serialized_start=314
  _globals['_UPDATEWEIGHTRSP']._serialized_end=382
  _globals['_PREDICTORSERVICE']._serialized_start=385
  _globals['_PREDICTORSERVICE']._serialized_end=540
# @@protoc_insertion_point(module_scope)
