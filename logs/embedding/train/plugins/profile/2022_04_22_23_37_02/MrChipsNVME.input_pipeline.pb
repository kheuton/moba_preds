	-AF@?#@-AF@?#@!-AF@?#@	?M?Ĭ???M?Ĭ??!?M?Ĭ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$-AF@?#@ͰQ?o&??A?????"@Y? ???۲?*	A`??"CS@2U
Iterator::Model::ParallelMapV2V?@?)V??!ȿg?B@)V?@?)V??1ȿg?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?n??S??!?b0E??2@)?&1???1h?\?C?0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceMg'?????!3L:???+@)Mg'?????13L:???+@:Preprocessing2F
Iterator::Modelo~?D???!?'?<`sI@)??b????1잱??o+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<??)t^??!}?ß?H@)[|
??z?1??K,L? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate~t??gy??!@???O3@)a???)q?1?D????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap o?ŏ??!,^?*1B6@)?a???b?1b7?Mő@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????Y?!:l??"M @)?????Y?1:l??"M @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?M?Ĭ??Ie??v??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ͰQ?o&??ͰQ?o&??!ͰQ?o&??      ??!       "      ??!       *      ??!       2	?????"@?????"@!?????"@:      ??!       B      ??!       J	? ???۲?? ???۲?!? ???۲?R      ??!       Z	? ???۲?? ???۲?!? ???۲?b      ??!       JCPU_ONLYY?M?Ĭ??b qe??v??X@