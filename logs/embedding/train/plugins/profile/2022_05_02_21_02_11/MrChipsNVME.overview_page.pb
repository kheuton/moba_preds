?	?V횈(@?V횈(@!?V횈(@	)??m%!@)??m%!@!)??m%!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?V횈(@D?;??)??A??խ?3'@Y??
G????*	?????`?@2F
Iterator::Model?drjg???!v?ۆ?W@)@7n1???1????˔P@:Preprocessing2U
Iterator::Model::ParallelMapV2Þv?k???!?iLj??;@)Þv?k???1?iLj??;@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?*5{???!?G??l?@)?*5{???1?G??l?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatˡE?????!??2?J??)]????ہ?1?f??????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ȓ?k&??!??N?7@)?????%n?1Q?Gsx??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???VC???!?????@)??m?2k?1?Ѳ?cE??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoroӟ?HY?!p??????)oӟ?HY?1p??????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??*????!PО??+	@)?????W?1????7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9(??m%!@I?$!???W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	D?;??)??D?;??)??!D?;??)??      ??!       "      ??!       *      ??!       2	??խ?3'@??խ?3'@!??խ?3'@:      ??!       B      ??!       J	??
G??????
G????!??
G????R      ??!       Z	??
G??????
G????!??
G????b      ??!       JCPU_ONLYY(??m%!@b q?$!???W@Y      Y@q?}W?\?@"?
device?Your program is NOT input-bound because only 4.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 