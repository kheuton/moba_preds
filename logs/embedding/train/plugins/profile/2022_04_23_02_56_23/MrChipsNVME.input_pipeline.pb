	?&?|?k@@?&?|?k@@!?&?|?k@@	["pH.??["pH.??!["pH.??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?&?|?k@@7?xͫ:??A*A*?D@@YN??1??*X9?Ȫk@)       =2U
Iterator::Model::ParallelMapV2$c??յ?!w?JPKDC@)$c??յ?1w?JPKDC@:Preprocessing2F
Iterator::Model????\???!???K?:R@)6 B\9{??1f?DG?0A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenaterP?L۟?!Ҏ?0i,@)?L??~ޔ?1????Tj"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR`L8??!;??.͟@)X???<???1
F??jZ@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J?8????!0I?(d@)?J?8????10I?(d@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM?O???!旳??@2@)?Op????1?A????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<ۤ????!FX?о;@)?aodn?1????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensord?????]?!????+??)d?????]?1????+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9["pH.??I?oѣ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7?xͫ:??7?xͫ:??!7?xͫ:??      ??!       "      ??!       *      ??!       2	*A*?D@@*A*?D@@!*A*?D@@:      ??!       B      ??!       J	N??1??N??1??!N??1??R      ??!       Z	N??1??N??1??!N??1??b      ??!       JCPU_ONLYY["pH.??b q?oѣ?X@