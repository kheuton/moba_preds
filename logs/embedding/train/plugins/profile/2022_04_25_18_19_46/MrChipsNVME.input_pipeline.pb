	Z??!??B@Z??!??B@!Z??!??B@	b0_C???b0_C???!b0_C???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Z??!??B@
?B?Գ??A????czB@Y???ek??*	???S?@?@2F
Iterator::Model>?ͨy??!???m?W@)?%?<??1??/?V@:Preprocessing2U
Iterator::Model::ParallelMapV2ݲC?Ö??!?I?&7?@)ݲC?Ö??1?I?&7?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7??????!???^k?@)????o??19#%??>@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceW?}W???!?4ݶ??)W?}W???1?4ݶ??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?4)?^??!(nw'@)??ek}q?1?8?"0??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0?AC???!h??%	@)??ǘ??p?1o?[?3??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~oӟ?Ha?!h3T,????)~oӟ?Ha?1h3T,????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+??????!??M?@)?'?>?Y?1zv??b
??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9b0_C???Ix>??2?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
?B?Գ??
?B?Գ??!
?B?Գ??      ??!       "      ??!       *      ??!       2	????czB@????czB@!????czB@:      ??!       B      ??!       J	???ek?????ek??!???ek??R      ??!       Z	???ek?????ek??!???ek??b      ??!       JCPU_ONLYYb0_C???b qx>??2?X@