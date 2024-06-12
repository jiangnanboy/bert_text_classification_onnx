### bert text classification using onnx of(bert,albert,roberta,macbert and so on)
利用bert，albert,roberta,macbert等的onnx进行文本分类推理

-----------------------------------------------------------------------
将bert类模型转为onnx格式，利用java进行推理。

The bert model is converted to onnx format, and java is used for inference.

* 模型转为onnx见https://github.com/jiangnanboy/model2onnx
* 测试模型来自https://huggingface.co/thu-coai/roberta-base-cold

### usage
【src/main/java/text_classification/TextClssify】

``` java
load model and vocab:
    static {
        tokenizer = new BertTokenizer("D:\\project\\idea_workspace\\bert_text_classification_onnx\\src\\main\\resources\\vocab.txt");
        try {
            LoadModel.loadModel("D:\\project\\idea_workspace\\bert_text_classification_onnx\\src\\main\\resources\\model.onnx");
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
 
pred:
        String text = "黑人很多都好吃懒做，偷奸耍滑！";
        Map<String, OnnxTensor> onnxTensorMap = null;
        try {
            onnxTensorMap = parseInputText(text);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Pair<Integer, Float> maskPredictions = pred(onnxTensorMap);
        System.out.println(maskPredictions);

result:
init bertTokenizer...
load vocab ...
load model...
[[CLS], 黑, 人, 很, 多, 都, 好, 吃, 懒, 做, ，, 偷, 奸, 耍, 滑, ！, [SEP]]
1
0.9989349
(1,0.9989349)
```

### requirement
java11+

onnxruntime1.11.0

### contact
- github：https://github.com/jiangnanboy

### reference
- https://github.com/jiangnanboy/model2onnx
- https://github.com/jiangnanboy/java_textcnn_onnx
- https://github.com/jiangnanboy/ad_detection
- https://huggingface.co/thu-coai/roberta-base-cold

