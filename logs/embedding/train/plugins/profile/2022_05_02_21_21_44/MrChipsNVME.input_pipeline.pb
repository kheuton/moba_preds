	_????'@_????'@!_????'@	L)x????L)x????!L)x????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$_????'@?	?????AdT8:'@YA???N??*	??~j??Y@2U
Iterator::Model::ParallelMapV2??9#J{??!??V???B@)??9#J{??1??V???B@:Preprocessing2F
Iterator::Modeluw??g??!	t\ɉ?Q@)?-??T??1	b??@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatN???????!=?,ܡ)@)?*5{???1e0?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice<k?]h???!eBE???"@)<k?]h???1eBE???"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*:??H??!?/????=@)S?!?uqk?1?h~??(
@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???m??!}X??(@)N^??i?1`PK?q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorQf?L2rV?!`eae??)Qf?L2rV?1`eae??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMappw?n?Ќ?!G!럢w+@)
K<?l?U?1LV??t???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9L)x????Ih?0?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?	??????	?????!?	?????      ??!       "      ??!       *      ??!       2	dT8:'@dT8:'@!dT8:'@:      ??!       B      ??!       J	A???N??A???N??!A???N??R      ??!       Z	A???N??A???N??!A???N??b      ??!       JCPU_ONLYYL)x????b qh?0?X@