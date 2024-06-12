package text_classification;

import ai.onnxruntime.*;
import org.apache.commons.lang3.tuple.Pair;
import text_classification.bert_tokenizer.tokenizerimpl.BertTokenizer;
import text_classification.bert_tokenizer.LoadModel;
import util.CollectionUtil;

import java.util.List;
import java.util.Map;

public class TextClassify {

    static BertTokenizer tokenizer;
    public static void main(String[] args) {
        String text = "黑人很多都好吃懒做，偷奸耍滑！";
        Map<String, OnnxTensor> onnxTensorMap = null;
        try {
            onnxTensorMap = parseInputText(text);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Pair<Integer, Float> maskPredictions = pred(onnxTensorMap);
        System.out.println(maskPredictions);
    }

    static {
        tokenizer = new BertTokenizer("resources\\vocab.txt");
        try {
            LoadModel.loadModel("resources\\model.onnx");
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    /**
     * tokenize text
     * @param text
     * @return
     * @throws Exception
     */
    public static Map<String, OnnxTensor> parseInputText(String text) throws Exception{
        OrtEnvironment env = LoadModel.env;
        List<String > tokens = tokenizer.tokenize(text);

        System.out.println(tokens);


        List<Integer> tokenIds = tokenizer.convert_tokens_to_ids(tokens);
        long[] inputIds = new long[tokenIds.size()];
        long[] attentionMask = new long[tokenIds.size()];
        long[] tokenTypeIds = new long[tokenIds.size()];
        for(int index=0; index < tokenIds.size(); index ++) {
            inputIds[index] = tokenIds.get(index);
            attentionMask[index] = 1;
            tokenTypeIds[index] = 0;
        }
        long[] shape = new long[]{1, inputIds.length};
        Object ObjInputIds = OrtUtil.reshape(inputIds, shape);
        Object ObjAttentionMask = OrtUtil.reshape(attentionMask, shape);
        Object ObjTokenTypeIds = OrtUtil.reshape(tokenTypeIds, shape);
        OnnxTensor input_ids = OnnxTensor.createTensor(env, ObjInputIds);
        OnnxTensor attention_mask = OnnxTensor.createTensor(env, ObjAttentionMask);
        OnnxTensor token_type_ids = OnnxTensor.createTensor(env, ObjTokenTypeIds);
        Map inputs = Map.of("input_ids", input_ids, "attention_mask", attention_mask, "token_type_ids", token_type_ids);
        return inputs;
    }


    public static Pair<Integer, Float> pred(Map<String, OnnxTensor> onnxTensorMap) {
        Pair<Integer, Float> pairResult = null;
        try{
            OrtSession session = LoadModel.session;
            try(OrtSession.Result results = session.run(onnxTensorMap)) {
                OnnxValue onnxValue = results.get(0);
                float[][] labels = (float[][]) onnxValue.getValue();
                float[] maskLables = labels[0];
                maskLables = softmax(maskLables);
                pairResult = predMax(maskLables);
                System.out.println(pairResult.getLeft());
                System.out.println(pairResult.getRight());

            }
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return pairResult;
    }

    public static Pair<Integer, Float> predMax(float[] probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
                idx = i;
            }
        }
        return Pair.of(idx, maxVal);
    }

    public static float[] softmax(float[] input) {
        List<Float> inputList = CollectionUtil.newArrayList();
        for(int i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }
        double inputSum = inputList.stream().mapToDouble(Math::exp).sum();
        float[] output = new float[input.length];
        for (int i=0; i<input.length; i++) {
            output[i] = (float) (Math.exp(input[i]) / inputSum);
        }
        return output;
    }

}

