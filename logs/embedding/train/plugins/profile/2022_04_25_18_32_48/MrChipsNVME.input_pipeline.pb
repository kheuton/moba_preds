	?U,~SP-@?U,~SP-@!?U,~SP-@	???Z??????Z???!???Z???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?U,~SP-@?c${????A?9?m??,@Y???????*	`??"۱^@2U
Iterator::Model::ParallelMapV2??'?Ȥ?!????@@)??'?Ȥ?1????@@:Preprocessing2F
Iterator::Model?:?*???!)?"/?XP@)???g?R??1jd3??)@@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice!?> ?M??!?k???&@)!?> ?M??1?k???&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatP?c*???!w?x3?\)@)??J?8??1????cr&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??(\?¥?!????NA@)?;?D~?1y?7f@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?-?????!WN???,@)?Ws?`?n?1?????M@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???3K??!??7]?/@)??hUM`?1_??>????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?n??S]?!???/?S??)?n??S]?1???/?S??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???Z???IRJ???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?c${?????c${????!?c${????      ??!       "      ??!       *      ??!       2	?9?m??,@?9?m??,@!?9?m??,@:      ??!       B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JCPU_ONLYY???Z???b qRJ???X@