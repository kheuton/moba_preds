	I?L?پ @I?L?پ @!I?L?پ @	
??u???
??u???!
??u???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$I?L?پ @?b?????AyY\ @Y????=z??*	F????xW@2U
Iterator::Model::ParallelMapV2?^Cp\ƥ?!?+?b?F@)?^Cp\ƥ?1?+?b?F@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?B??Ð?!+@#Bo1@)*?	??$??1ʚ???Z/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?N^???!=ư??&@)?N^???1=ư??&@:Preprocessing2F
Iterator::Modele?P3????!*d&,Y?K@)??4??1v??'?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???Or???!՛?ӦdF@)"q??]??1ұ?	1!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??.Q?5??!?7@?J?0@)<?.9?t?16et_G?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn??S??!?S??r?2@)~t??gy^?1??1?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?[?!gܣ/]??)F%u?[?1gܣ/]??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9
??u???It?+??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b??????b?????!?b?????      ??!       "      ??!       *      ??!       2	yY\ @yY\ @!yY\ @:      ??!       B      ??!       J	????=z??????=z??!????=z??R      ??!       Z	????=z??????=z??!????=z??b      ??!       JCPU_ONLYY
??u???b qt?+??X@