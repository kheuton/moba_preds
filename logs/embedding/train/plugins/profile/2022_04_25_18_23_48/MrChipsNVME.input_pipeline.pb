	$?6?D?.@$?6?D?.@!$?6?D?.@	ă??%??ă??%??!ă??%??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$$?6?D?.@mo?$???A6׆?q.@Y?"?4???*	????x?_@2F
Iterator::Model6??,
???!??z?)?O@)??
}????1??l?@@:Preprocessing2U
Iterator::Model::ParallelMapV2?jH?c???!?u???>@)?jH?c???1?u???>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??:???!Im?J*+@)P?c*???1????}(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???C???!???d&@)???C???1???d&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??-?v???!*?P?'B@)a??_Yi??1_3?I1H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateB???8a??!FU익;,@)*??g\8p?1?F??t?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap͑?_c??!?i:??P/@)X??j`?1C?p???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?[?!#T?sd??)_?Q?[?1#T?sd??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ã??%??I??H???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	mo?$???mo?$???!mo?$???      ??!       "      ??!       *      ??!       2	6׆?q.@6׆?q.@!6׆?q.@:      ??!       B      ??!       J	?"?4????"?4???!?"?4???R      ??!       Z	?"?4????"?4???!?"?4???b      ??!       JCPU_ONLYYÃ??%??b q??H???X@